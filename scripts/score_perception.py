"""
LLM Perception Evaluator
========================
Zero-shot scoring of visualization images for human perceptual properties
using Claude (Anthropic) or GPT (OpenAI).

Usage:
    python scripts/score_perception.py --perception memorability --provider claude
    python scripts/score_perception.py --perception vc --provider gpt
    python scripts/score_perception.py --perception memorability --provider claude --model claude-sonnet-4-6 --concurrency 5
    python scripts/score_perception.py --perception vc --provider gpt --model gpt-5.4 --limit 20

Outputs to:  results/<perception>/<perception>_<model>_scores.csv
Repeated runs use: results/<perception>/runs/<perception>_<model>_run<N>_scores.csv
Resume-safe: already-scored images are skipped unless --overwrite is passed.
"""

import os, sys, json, time, argparse, base64, csv, asyncio, threading
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

SCRIPT_DIR  = Path(__file__).parent
CONFIG_DIR  = SCRIPT_DIR.parent / 'config'
RESULTS_DIR = SCRIPT_DIR.parent / 'results'

MAX_TOKENS           = 800
SLEEP_BETWEEN        = 0.3          # seconds between sequential calls (unused in async mode)
DEFAULT_CLAUDE_MODEL = 'claude-sonnet-4-6'
DEFAULT_GPT_MODEL    = 'gpt-5.4'

_csv_lock = threading.Lock()


def run_async_compat(coro):
    """Run a coroutine from sync code, even if a loop is already running (e.g., Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result = {'value': None, 'error': None}

    def _target():
        try:
            result['value'] = asyncio.run(coro)
        except Exception as e:
            result['error'] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()

    if result['error'] is not None:
        raise result['error']
    return result['value']


# ── Shared helpers ────────────────────────────────────────────────────────────

def get_media_type(filename: str) -> str:
    ext = Path(filename).suffix.lower().lstrip('.')
    return {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'gif': 'image/gif', 'webp': 'image/webp'}.get(ext, 'image/png')


MAX_PIXELS   = 4000              # max dimension sent to APIs
MAX_BYTES    = 10 * 1024 * 1024  # 10 MB hard limit (Claude)


def resize_if_needed(data: bytes) -> bytes:
    """Resize/compress image in-memory to fit dimension and size limits."""
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(data))
    needs_resize = max(img.size) > MAX_PIXELS
    needs_compress = len(data) > MAX_BYTES

    if not needs_resize and not needs_compress:
        return data

    # Resize if needed
    if needs_resize:
        ratio = MAX_PIXELS / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

    # Convert to JPEG-compatible mode
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')

    # Compress as JPEG at decreasing quality until under MAX_BYTES
    for quality in (85, 75, 60, 45, 30):
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality, optimize=True)
        result = buf.getvalue()
        if len(result) <= MAX_BYTES:
            return result

    return result  # best effort


def detect_media_type(data: bytes) -> str:
    if data[:3] == b'\xff\xd8\xff':
        return 'image/jpeg'
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return 'image/gif'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    return get_media_type(data)  # fallback to extension-based guess


def load_image_base64(img_name: str, url: str) -> tuple[str, str] | tuple[None, None]:
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        data = resize_if_needed(data)
        return base64.standard_b64encode(data).decode('utf-8'), detect_media_type(data)
    except Exception as e:
        print(f'  WARNING: failed to fetch {img_name}: {e}')
        return None, None


def parse_response(raw: str) -> dict:
    text = raw.strip()
    if text.startswith('```'):
        parts = text.split('```', 2)
        text = parts[1]
        if text.startswith('json'):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def load_existing_scores(scores_csv: Path) -> set:
    if not scores_csv.exists():
        return set()
    with open(scores_csv, 'r', newline='', encoding='utf-8') as f:
        return {
            row['imageName']
            for row in csv.DictReader(f)
            if row.get('imageName') and row.get('score', '').strip()
        }


def append_row(csv_path: Path, row: dict):
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, 'a+', newline='', encoding='utf-8') as f:
        if write_header:
            fieldnames = list(row.keys())
        else:
            f.seek(0)
            fieldnames = next(csv.reader(f))
            f.seek(0, os.SEEK_END)
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


# ── Claude async scoring ───────────────────────────────────────────────────────

async def _score_one_claude(aclient, sem, idx, total, config, img_name, img_url, scores_csv, model, run_id):
    import anthropic
    async with sem:
        b64, media_type = load_image_base64(img_name, img_url)
        if b64 is None:
            print(f'[{idx}/{total}] {img_name} ... SKIP (fetch failed)')
            return (img_name, None)
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'source': {'type': 'base64',
                                         'media_type': media_type,
                                         'data': b64}},
            {'type': 'text', 'text': config['user_message']}
        ]}]
        try:
            response = await aclient.messages.create(
                model=model, messages=messages, max_tokens=MAX_TOKENS,
                temperature=0,
                system=[{'type': 'text', 'text': config['system_prompt']}]
            )
            raw    = next(b.text for b in response.content if b.type == 'text')
            result = parse_response(raw)
            score      = result.get(config['score_key'])
            confidence = result.get(config['confidence_key'])
            with _csv_lock:
                append_row(scores_csv, {
                    'imageName':            img_name,
                    'run_id':               run_id,
                    'score':                score,
                    'confidence':           confidence,
                    'explanation':          result.get('explanation', ''),
                    'chart_type':           result.get('chart_type', ''),
                    'chart_type_confidence': result.get('chart_type_confidence', ''),
                })
            print(f'[{idx}/{total}] {img_name} ... {config["score_key"]}={score}')
            return (img_name, score)
        except anthropic.RateLimitError:
            print(f'[{idx}/{total}] {img_name} ... RATE LIMITED — waiting 60 s')
            await asyncio.sleep(60)
            return (img_name, None)
        except Exception as e:
            print(f'[{idx}/{total}] {img_name} ... ERROR: {e}')
            return (img_name, None)


async def run_claude(api_key, model, config, to_process, scores_csv, concurrency, run_id):
    import anthropic
    aclient = anthropic.AsyncAnthropic(api_key=api_key)
    sem     = asyncio.Semaphore(concurrency)
    total   = len(to_process)
    tasks   = [
        _score_one_claude(aclient, sem, i, total, config,
                          r['imageName'], r['imageURL'], scores_csv, model, run_id)
        for i, r in enumerate(to_process, 1)
    ]
    results = await asyncio.gather(*tasks)
    ok      = sum(1 for _, s in results if s is not None)
    failed  = [n for n, s in results if s is None]
    return ok, failed


# ── GPT async scoring ──────────────────────────────────────────────────────────

async def _score_one_gpt(aclient, sem, idx, total, config, img_name, img_url, scores_csv, model, run_id):
    from openai import RateLimitError
    async with sem:
        b64, media_type = load_image_base64(img_name, img_url)
        if b64 is None:
            print(f'[{idx}/{total}] {img_name} ... SKIP (fetch failed)')
            return (img_name, None)
        messages = [
            {'role': 'system', 'content': config['system_prompt']},
            {'role': 'user', 'content': [
                {'type': 'image_url',
                 'image_url': {'url': f'data:{media_type};base64,{b64}', 'detail': 'high'}},
                {'type': 'text', 'text': config['user_message']}
            ]}
        ]
        try:
            response = await aclient.chat.completions.create(
                model=model, messages=messages,
                max_completion_tokens=MAX_TOKENS, temperature=0
            )
            raw    = response.choices[0].message.content
            result = parse_response(raw)
            score      = result.get(config['score_key'])
            confidence = result.get(config['confidence_key'])
            with _csv_lock:
                append_row(scores_csv, {
                    'imageName':            img_name,
                    'run_id':               run_id,
                    'score':                score,
                    'confidence':           confidence,
                    'explanation':          result.get('explanation', ''),
                    'chart_type':           result.get('chart_type', ''),
                    'chart_type_confidence': result.get('chart_type_confidence', ''),
                })
            print(f'[{idx}/{total}] {img_name} ... {config["score_key"]}={score}')
            return (img_name, score)
        except RateLimitError:
            print(f'[{idx}/{total}] {img_name} ... RATE LIMITED — waiting 60 s')
            await asyncio.sleep(60)
            return (img_name, None)
        except Exception as e:
            print(f'[{idx}/{total}] {img_name} ... ERROR: {e}')
            return (img_name, None)


async def run_gpt(api_key, model, config, to_process, scores_csv, concurrency, run_id):
    from openai import AsyncOpenAI
    aclient = AsyncOpenAI(api_key=api_key)
    sem     = asyncio.Semaphore(concurrency)
    total   = len(to_process)
    tasks   = [
        _score_one_gpt(aclient, sem, i, total, config,
                       r['imageName'], r['imageURL'], scores_csv, model, run_id)
        for i, r in enumerate(to_process, 1)
    ]
    results = await asyncio.gather(*tasks)
    ok      = sum(1 for _, s in results if s is not None)
    failed  = [n for n, s in results if s is None]
    return ok, failed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Zero-shot LLM scoring of visualization images for perceptual properties.'
    )
    parser.add_argument('--perception',  required=True,
                        help='Perception type — must match a file in config/<name>.json')
    parser.add_argument('--provider',    required=True, choices=['claude', 'gpt'],
                        help='LLM provider')
    parser.add_argument('--model',       type=str, default=None,
                        help=f'Model name (defaults: claude→{DEFAULT_CLAUDE_MODEL}, '
                             f'gpt→{DEFAULT_GPT_MODEL})')
    parser.add_argument('--concurrency', type=int, default=5,
                        help='Number of parallel API calls (default: 5)')
    parser.add_argument('--limit',       type=int, default=None,
                        help='Process at most N images (useful for testing)')
    parser.add_argument('--overwrite',   action='store_true',
                        help='Re-score images that already have results')
    parser.add_argument('--run-id',      type=int, choices=[2, 3], default=None,
                        help='Store a repeated run under results/<perception>/runs '
                             '(use 2 or 3; omit for the existing run-1 file)')
    parser.add_argument('--outdir',      type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--data-csv',    type=str, default=None, dest='data_csv',
                        help='Override the image list CSV from the config '
                             '(keeps the same output file, useful for topping up predictions)')
    args = parser.parse_args()

    # ── Load perception config ──────────────────────────────────────────────
    cfg_path = CONFIG_DIR / f'{args.perception}.json'
    if not cfg_path.exists():
        print(f'ERROR: config not found: {cfg_path}')
        print(f'  Available: {[p.stem for p in CONFIG_DIR.glob("*.json")]}')
        sys.exit(1)
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # ── Load image list ─────────────────────────────────────────────────────
    data_csv = Path(args.data_csv or config['data_csv'])
    if not data_csv.is_absolute():
        data_csv = SCRIPT_DIR.parent / data_csv
    if not data_csv.exists():
        print(f'ERROR: data CSV not found: {data_csv}')
        print('  Run  python data/prepare_data.py  first.')
        sys.exit(1)
    df = pd.read_csv(data_csv)
    df = df.rename(columns={
        config['image_col']: 'imageName',
        config['url_col']:   'imageURL',
    })
    all_images = df[['imageName', 'imageURL']].drop_duplicates('imageName').to_dict('records')

    # ── Resolve output path ─────────────────────────────────────────────────
    provider  = args.provider
    model     = args.model or (DEFAULT_CLAUDE_MODEL if provider == 'claude' else DEFAULT_GPT_MODEL)
    model_tag = model.replace('/', '-')
    outdir    = Path(args.outdir) if args.outdir else (RESULTS_DIR / args.perception)
    if args.run_id is not None:
        outdir = outdir / 'runs'
    outdir.mkdir(parents=True, exist_ok=True)
    run_suffix = f'_run{args.run_id}' if args.run_id is not None else ''
    scores_csv = outdir / f'{args.perception}_{model_tag}{run_suffix}_scores.csv'

    # ── Resume / overwrite ──────────────────────────────────────────────────
    if args.overwrite and scores_csv.exists():
        scores_csv.unlink()
    done       = set() if args.overwrite else load_existing_scores(scores_csv)
    to_process = [r for r in all_images if r['imageName'] not in done]
    if args.limit:
        to_process = to_process[:args.limit]

    print(f'Perception  : {args.perception}')
    print(f'Provider    : {provider}  |  Model: {model}')
    print(f'To process  : {len(to_process)} images  ({len(done)} already done, '
          f'{len(all_images)} total)')
    print(f'Concurrency : {args.concurrency}')
    print(f'Output      : {scores_csv}\n')

    if not to_process:
        print('Nothing to process.  Pass --overwrite to re-score.')
        return

    # ── Run ─────────────────────────────────────────────────────────────────
    if provider == 'claude':
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print('ERROR: ANTHROPIC_API_KEY not set in environment or .env')
            sys.exit(1)
        ok, failed = run_async_compat(
            run_claude(api_key, model, config, to_process, scores_csv,
                       args.concurrency, args.run_id or 1)
        )
    else:  # gpt
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print('ERROR: OPENAI_API_KEY not set in environment or .env')
            sys.exit(1)
        ok, failed = run_async_compat(
            run_gpt(api_key, model, config, to_process, scores_csv,
                    args.concurrency, args.run_id or 1)
        )

    print(f'\nFinished: {ok} scored, {len(failed)} failed')
    if failed:
        print(f'Failed images ({len(failed)}): {failed[:10]}{"..." if len(failed) > 10 else ""}')


if __name__ == '__main__':
    main()
