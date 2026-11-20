import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from healthrolebench.common.artifacts import private_text_artifact
from healthrolebench.common.items import CONDITIONS, default_output, load_items
from healthrolebench.common.labels import INSTRUCTIONS

# Override with HRB_HF_CACHE on clusters (for example, on HPC clusters) where the home cache is unavailable.
CACHE = Path(os.environ.get('HRB_HF_CACHE', Path.home() / '.cache' / 'huggingface' / 'hub'))
MAX_NEW_TOKENS = 48  # v1 used 14, which truncated LLaMA-2 before it named a label
MODELS = {
    'llama2': ('meta-llama--Llama-2-7b-chat-hf', 'f5db02db724555f92da89c216ac04704f23d4590'),
    'mistral': ('mistralai--Mistral-7B-Instruct-v0.2', '63a8b081895390a26e140280378bc85ec8bce07a'),
    'qwen': ('Qwen--Qwen3.5-2B', '15852e8c16360a2fea060d615a32b45270f8a8fc'),
    'gemma': ('google--gemma-7b', 'ff6768d9368919a1f025a54f9f5aa0ee591730bb'),
}


def run(name, condition='original', output_dir=None, max_new_tokens=MAX_NEW_TOKENS, items='audit'):
    folder, revision = MODELS[name]
    cache = Path(os.environ.get('HRB_GEMMA_CACHE', 'models')) if name == 'gemma' else CACHE
    snapshot = cache / ('models--' + folder) / 'snapshots' / revision
    if not (snapshot / 'config.json').is_file():
        raise FileNotFoundError(snapshot)
    output_dir = output_dir or default_output(items, condition)
    out = Path('data/experiments') / output_dir
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'{name}.csv'
    if target.exists():
        raise FileExistsError(target)
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, padding_side='left')
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    model_class = AutoModelForImageTextToText if config.model_type == 'qwen3_5' else AutoModelForCausalLM
    model = model_class.from_pretrained(snapshot, local_files_only=True, dtype=torch.float16).to('cuda:0').eval()
    id_column, rows = load_items(items, condition)
    with private_text_artifact(target) as handle:
        writer = csv.writer(handle)
        writer.writerow((id_column, 'raw_completion'))
        for start in range(0, len(rows), 12):
            batch = rows[start:start+12]
            prompts = [INSTRUCTIONS + json.dumps(post, ensure_ascii=False) + chr(10) + 'Category:' for _, post in batch]
            if name != 'gemma':
                prompts = [tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False) for prompt in prompts]
            inputs = tok(prompts, padding=True, return_tensors='pt', add_special_tokens=(name == 'gemma')).to('cuda:0')
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
            responses = tok.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            if len(responses) != len(batch):
                raise ValueError('Generation count differs from input batch')
            writer.writerows((item, response) for (item, _), response in zip(batch, responses))
            handle.flush()
            if start % 120 == 0:
                print(name, min(start+12, len(rows)), '/', len(rows), flush=True)
    settings = dict(model=folder, revision=revision, items=items, n=len(rows), condition=condition, max_new_tokens=max_new_tokens,
                    decoding='greedy', batch_size=12, instructions_sha256=hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    (out / f'{name}_settings.json').write_text(json.dumps(settings, indent=2) + chr(10), encoding='utf-8')
    print('Completed:', name, len(rows), revision, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot classification of the audit sample or the evaluation set (run from PROJECT_ROOT).')
    parser.add_argument('model', choices=MODELS)
    parser.add_argument('--condition', default='original', choices=CONDITIONS)
    parser.add_argument('--items', default='audit', choices=('audit', 'eval'))
    parser.add_argument('--output-dir', help='Run directory under data/experiments (default depends on --items and --condition)')
    parser.add_argument('--max-new-tokens', type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()
    run(args.model, args.condition, args.output_dir, args.max_new_tokens, args.items)
