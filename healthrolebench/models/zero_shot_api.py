"""GPT-4o zero-shot classification of the frozen 450-post audit sample.

Mirrors healthrolebench.models.zero_shot (same instructions, same audit sample, same
original / cue-masked conditions) through the OpenAI Chat Completions API with
deterministic decoding (temperature 0, top_p 1). Output has the same schema
(audit_id, raw_completion), so pilot_tables and stress_tables score it with the
shared labels-v2 parser.

The API key is read from OPENAI_API_KEY or --key-file and never logged.
Spending is bounded by --max-usd; usage and cost are recorded per run.
"""
import argparse
import csv
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from healthrolebench.common.artifacts import private_text_artifact
from healthrolebench.common.items import CONDITIONS, default_output, load_items
from healthrolebench.common.labels import INSTRUCTIONS

MODEL = 'gpt-4o-2024-08-06'          # same dated snapshot as the generator
PRICE_PER_MTOK = {'input': 2.50, 'output': 10.00}   # USD per million tokens for this snapshot
MAX_TOKENS = 48                      # matches the open-model answer budget
URL = 'https://api.openai.com/v1/chat/completions'


def call(key, prompt, retries=4):
    body = dict(model=MODEL, messages=[{'role': 'user', 'content': prompt}], temperature=0, top_p=1,
                max_tokens=MAX_TOKENS, seed=42)
    request = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(f'OpenAI API error {error.code}') from None
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise


def run(condition, output_dir, max_usd, key, items='audit'):
    out = Path('data/experiments') / (output_dir or default_output(items, condition))
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'gpt4o.csv'
    if target.exists():
        raise FileExistsError(target)
    id_column, rows = load_items(items, condition)
    usage = {'prompt_tokens': 0, 'completion_tokens': 0}
    models, fingerprints, cost = set(), set(), 0.0
    with private_text_artifact(target) as handle:
        writer = csv.writer(handle)
        writer.writerow((id_column, 'raw_completion'))
        for i, (item, post) in enumerate(rows, 1):
            # Same prompt text the open chat models receive (before their chat template).
            prompt = INSTRUCTIONS + json.dumps(post, ensure_ascii=False) + chr(10) + 'Category:'
            data = call(key, prompt)
            choice = data['choices'][0]
            writer.writerow((item, choice['message'].get('content') or ''))
            for k in usage:
                usage[k] += data.get('usage', {}).get(k, 0)
            models.add(data.get('model', ''))
            fingerprints.add(data.get('system_fingerprint') or '')
            cost = (usage['prompt_tokens'] * PRICE_PER_MTOK['input'] + usage['completion_tokens'] * PRICE_PER_MTOK['output']) / 1e6
            if cost > max_usd:
                raise RuntimeError(f'Spending limit reached after {i} posts (US${cost:.4f}); no output published')
            if i % 50 == 0:
                print('gpt4o', condition, i, '/', len(rows), f'US${cost:.4f}', flush=True)
    settings = dict(model=MODEL, items=items, n=len(rows), response_models=sorted(models), system_fingerprints=sorted(fingerprints),
                    condition=condition, temperature=0, top_p=1, max_tokens=MAX_TOKENS, seed=42,
                    instructions_sha256=hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    usage=usage, price_per_mtok=PRICE_PER_MTOK, cost_usd=round(cost, 6),
                    note='API decoding is not guaranteed bit-identical across calls; raw completions are kept.')
    (out / 'gpt4o_settings.json').write_text(json.dumps(settings, indent=2) + chr(10), encoding='utf-8')
    print('Completed: gpt4o', condition, len(rows), f'US${cost:.4f}', flush=True)
    return cost


def main():
    parser = argparse.ArgumentParser(description='GPT-4o zero-shot classification of the audit sample (run from PROJECT_ROOT).')
    parser.add_argument('--condition', default='original', choices=CONDITIONS)
    parser.add_argument('--items', default='audit', choices=('audit', 'eval'))
    parser.add_argument('--output-dir')
    parser.add_argument('--max-usd', type=float, default=1.0, help='Abort before publishing if this run would cost more')
    parser.add_argument('--key-file', type=Path, help='File containing only the API key (else OPENAI_API_KEY)')
    args = parser.parse_args()
    key = args.key_file.read_text().strip() if args.key_file else os.environ.get('OPENAI_API_KEY', '')
    if not key:
        parser.error('Set OPENAI_API_KEY or pass --key-file')
    run(args.condition, args.output_dir, args.max_usd, key, args.items)


if __name__ == '__main__':
    main()
