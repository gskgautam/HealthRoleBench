"""Full-data frozen-backbone classifiers (Tables 8 and 10).

For each open model (LLaMA-2, Mistral, Qwen, Gemma):
1. `embed`: mean-pool the final hidden layer (attention-masked, max 256 tokens) for all
   60,000 posts once, cached as float16 under data/experiments/probe_full_v1/embeddings/.
2. `heads`: for every held-out generator G != the classifier, train a linear softmax head on
   the four other generators and test it on all 12,000 posts of G (Table 8) and on G's
   300 evaluation-set posts (paired with zero-shot in Table 10).

Choices (fixed, documented for the paper):
- Leakage: training posts whose leakage group touches any post of G are removed.
- Class imbalance: cross-entropy weighted by inverse class frequency (Macro-F1 is primary).
- Optimisation: full-batch AdamW, lr 0.01, weight decay 0.01, features L2-normalised.
  The number of steps (50..1000, every 50) is chosen on a 10% group-disjoint validation
  split of the training generators; the test generator is never used for selection.
- Seeds 42, 43, 44 (head initialisation and validation split). The diagonal (a model
  classifying its own generator's posts) is excluded, as in the manuscript.

Run from PROJECT_ROOT:
  python -m healthrolebench.models.probe_full embed --project-root . --model llama2
  python -m healthrolebench.models.probe_full heads --project-root .
"""
import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from healthrolebench.common.data import MODELS as GENERATORS, digest, load_corpus
from healthrolebench.curation.eval_set import load_eval

OUT = Path('data/experiments/probe_full_v1')
CLASSIFIERS = ('llama2', 'mistral', 'qwen', 'gemma')
SEEDS = (42, 43, 44)
MAX_STEPS, EVAL_EVERY = 1000, 50


def embed(root, model):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    from healthrolebench.models.zero_shot import CACHE, MODELS

    out = root / OUT / 'embeddings'
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'{model}.npy'
    if target.exists():
        raise FileExistsError(target)
    records, provenance = load_corpus(root)
    folder, revision = MODELS[model]
    cache = Path(os.environ.get('HRB_GEMMA_CACHE', root / 'models')) if model == 'gemma' else CACHE
    snapshot = cache / ('models--' + folder) / 'snapshots' / revision
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    model_class = AutoModelForImageTextToText if config.model_type == 'qwen3_5' else AutoModelForCausalLM
    backbone = model_class.from_pretrained(snapshot, local_files_only=True, dtype=torch.float16).to('cuda:0').eval()
    base = backbone.model.language_model if config.model_type == 'qwen3_5' else backbone.model
    order = sorted(range(len(records)), key=lambda i: len(records[i].text))   # length-sorted batches
    vectors = np.zeros((len(records), base.config.hidden_size), dtype=np.float16)
    bf16_batches = []
    with torch.inference_mode():
        for start in range(0, len(order), 32):
            index = order[start:start + 32]
            batch = tokenizer([records[i].text for i in index], padding=True, truncation=True,
                              max_length=256, return_tensors='pt').to('cuda:0')
            mask = batch.attention_mask.unsqueeze(-1)
            states = base(**batch, use_cache=False, return_dict=True).last_hidden_state.float()
            pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            if not torch.isfinite(pooled).all():
                # float16 overflow (known for Gemma): redo this batch in bfloat16 and record it.
                base.to(torch.bfloat16)
                states = base(**batch, use_cache=False, return_dict=True).last_hidden_state.float()
                base.to(torch.float16)
                pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                bf16_batches.append(start)
                if not torch.isfinite(pooled).all():
                    raise FloatingPointError(f'Non-finite embedding in batch starting {start}')
            vectors[index] = pooled.cpu().numpy().astype(np.float16)
            if start % 6400 == 0:
                print(model, 'embedded', min(start + 32, len(order)), '/', len(order), flush=True)
    temporary = out / f'.{model}.npy.tmp'
    with temporary.open('wb') as handle:
        np.save(handle, vectors)
    temporary.rename(target)
    with (out / f'{model}_ids.csv').open('w', newline='', encoding='utf-8') as handle:
        csv.writer(handle).writerows([('generator', 'sample_id')] + [(r.generator, r.sample_id) for r in records])
    (out / f'{model}_manifest.json').write_text(json.dumps(dict(
        model=folder, revision=revision, pooling='attention-masked mean of final hidden layer', max_tokens=256,
        dtype='float16', bf16_fallback_batches=bf16_batches, n=len(records), dim=int(vectors.shape[1]), corpus=provenance,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0), runner_sha256=digest(Path(__file__))), indent=2) + chr(10))
    print('Completed embeddings', model, vectors.shape, flush=True)


def macro_scores(true, pred, n_classes):
    true, pred = np.asarray(true), np.asarray(pred)
    f1, precision, recall = [], [], []
    for c in range(n_classes):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        precision.append(tp / (tp + fp) if tp + fp else 0.0)
        recall.append(tp / (tp + fn) if tp + fn else 0.0)
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1)), float(np.mean(precision)), float(np.mean(recall)), float((true == pred).mean())


def validation_mask(pool_groups, seed, fraction=.10, max_group=50):
    """Group-disjoint validation split of about `fraction` of the posts.

    Only groups of at most `max_group` posts are eligible, so one giant exemplar-linked
    group (REVIEW.md A3) can never move a whole generator-role cell out of training.
    """
    sizes = Counter(pool_groups.tolist())
    eligible = sorted(g for g, n in sizes.items() if n <= max_group)
    random.Random(seed).shuffle(eligible)
    chosen, total, target = set(), 0, fraction * len(pool_groups)
    for g in eligible:
        if total >= target:
            break
        chosen.add(g)
        total += sizes[g]
    return np.isin(pool_groups, list(chosen))


def train_head(torch, x, y, train_idx, val_idx, n_classes, seed):
    torch.manual_seed(seed)
    head = torch.nn.Linear(x.shape[1], n_classes, device=x.device)
    counts = torch.bincount(y[train_idx], minlength=n_classes).float()
    weight = counts.sum() / (n_classes * counts)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.01)
    best = (-1.0, 0, None)
    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(head(x[train_idx]), y[train_idx], weight=weight)
        loss.backward()
        optimizer.step()
        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                pred = head(x[val_idx]).argmax(dim=1).cpu().numpy()
            score = macro_scores(y[val_idx].cpu().numpy(), pred, n_classes)[0]
            if score > best[0]:
                best = (score, step, {k: v.detach().clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[2])
    return head, best[1], best[0]


def heads(root):
    import torch

    out = root / OUT
    for name in ('table_08_full.csv', 'predictions_full.csv', 'eval_predictions.csv'):
        if (out / name).exists():
            raise FileExistsError(out / name)
    records, provenance = load_corpus(root)
    key, _ = load_eval(root)
    labels = sorted({r.label for r in records})
    y_all = np.array([labels.index(r.label) for r in records])
    generator = np.array([r.generator for r in records])
    groups = np.array([r.group for r in records])
    position = {(r.generator, r.sample_id): i for i, r in enumerate(records)}
    eval_index = {item: position[row['generator'], row['sample_id']] for item, row in key.items()}
    available = [m for m in CLASSIFIERS if (out / 'embeddings' / f'{m}.npy').is_file()]
    if not available:
        raise SystemExit('No embeddings found; run the embed step first')
    summary, settings = [], {}
    with (out / 'predictions_full.csv').open('x', newline='', encoding='utf-8') as full_handle, \
         (out / 'eval_predictions.csv').open('x', newline='', encoding='utf-8') as eval_handle:
        full_writer, eval_writer = csv.writer(full_handle), csv.writer(eval_handle)
        full_writer.writerow(('classifier', 'held_generator', 'seed', 'sample_id', 'label', 'prediction'))
        eval_writer.writerow(('classifier', 'held_generator', 'seed', 'item_id', 'prediction'))
        for model in available:
            ids = [tuple(r) for r in csv.reader((out / 'embeddings' / f'{model}_ids.csv').open(encoding='utf-8'))][1:]
            if ids != [(r.generator, r.sample_id) for r in records]:
                raise ValueError(f'Embedding order differs from the corpus for {model}')
            x = torch.nn.functional.normalize(torch.from_numpy(np.load(out / 'embeddings' / f'{model}.npy')).float().cuda(), dim=1)
            y = torch.from_numpy(y_all).cuda()
            for held in GENERATORS:
                if held == model:
                    continue                                        # diagonal excluded
                test = np.flatnonzero(generator == held)
                touched = set(groups[test])
                pool = np.flatnonzero((generator != held) & ~np.isin(groups, list(touched)))
                held_eval = [(item, i) for item, i in eval_index.items() if generator[i] == held]
                scores = []
                for seed in SEEDS:
                    val_mask = validation_mask(groups[pool], seed)
                    train_idx = torch.from_numpy(pool[~val_mask]).cuda()
                    val_idx = torch.from_numpy(pool[val_mask]).cuda()
                    head, steps, val_f1 = train_head(torch, x, y, train_idx, val_idx, len(labels), seed)
                    with torch.no_grad():
                        pred = head(x[torch.from_numpy(test).cuda()]).argmax(dim=1).cpu().numpy()
                        eval_pred = head(x[torch.tensor([i for _, i in held_eval]).cuda()]).argmax(dim=1).cpu().numpy()
                    scores.append(macro_scores(y_all[test], pred, len(labels)))
                    full_writer.writerows((model, held, seed, records[i].sample_id, records[i].label, labels[p]) for i, p in zip(test, pred))
                    eval_writer.writerows((model, held, seed, item, labels[p]) for (item, _), p in zip(held_eval, eval_pred))
                    settings[f'{model}|{held}|{seed}'] = dict(train_n=int(len(train_idx)), validation_n=int(len(val_idx)),
                                                               purged=int((generator != held).sum() - len(pool)),
                                                               steps=steps, validation_macro_f1=val_f1)
                    print(model, 'held', held, 'seed', seed, 'steps', steps, 'test macro-F1', round(scores[-1][0], 4), flush=True)
                f1, precision, recall, accuracy = (np.array(column) for column in zip(*scores))
                summary.append((held, model, len(test), f1.mean(), f1.std(ddof=1), accuracy.mean(), precision.mean(), recall.mean()))
            del x
            torch.cuda.empty_cache()
    with (out / 'table_08_full.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('held_generator', 'classifier', 'n', 'macro_f1', 'macro_f1_sd', 'accuracy', 'macro_precision', 'macro_recall'))
        writer.writerows(summary)
    (out / 'heads_manifest.json').write_text(json.dumps(dict(
        labels=labels, seeds=SEEDS, optimizer='AdamW full batch', lr=.01, weight_decay=.01, max_steps=MAX_STEPS,
        step_selection='validation macro-F1 every 50 steps on a ~10% group-disjoint split (groups of <=50 posts) of the training generators',
        class_weighting='inverse frequency', leakage='training groups touching the held-out generator removed',
        classifiers=available, corpus=provenance, runs=settings, runner_sha256=digest(Path(__file__))), indent=2) + chr(10))
    print('Completed heads for', available, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full-data frozen-backbone classifiers (Tables 8 and 10).')
    parser.add_argument('step', choices=('embed', 'heads'))
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--model', choices=CLASSIFIERS)
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    root = args.project_root.resolve()
    if args.step == 'embed':
        if not args.model:
            parser.error('embed requires --model')
        embed(root, args.model)
    else:
        heads(root)
