"""Table 20: does HealthRoleBench help on real social-media posts? (domain adaptation)

Real data: CONSTRAINT@AAAI-2021 COVID-19 fake-news dataset (Patwa et al., 2021), 10,700 human-
labelled posts (real / fake) with fixed train / validation / test splits, stored under
data/external/constraint_covid19/ (source, revision and checksums in SOURCE.json, SHA256SUMS).

Steps (run from PROJECT_ROOT):
  pretrain  RoBERTa-base trained on all 60,000 HealthRoleBench posts (5 classes; group-disjoint
            ~20% validation split; same recipe as Table 12: 3 epochs, lr 1e-5, batch 32, best
            epoch by validation Macro-F1). Saved for reuse.
  transfer  Zero-shot transfer: the HealthRoleBench model labels the real test posts; any of the
            four misinformation roles counts as "fake", Otherwise as "real" (k = 0 row).
  finetune  Binary fine-tuning on k real training posts (k in 50, 100, 250, 500, 1000, all;
            class-balanced sample; seeds 42-44 control the sample and head initialisation) from
            two starting points: plain RoBERTa-base, or the HealthRoleBench model with a new
            2-class head. Up to 10 epochs (lr 2e-5, batch 16), best epoch on the full real
            validation split (identical for both starting points), scored once on the test split.
  table     Aggregates Macro-F1 / accuracy (mean, SD over seeds) and the gain from HealthRoleBench.
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np

from healthrolebench.common.data import digest, load_corpus
from healthrolebench.common.data import _train_validation

OUT = Path('data/experiments/domain_adapt_v1')
REAL = Path('data/external/constraint_covid19')
BASE = ('FacebookAI/roberta-base', 'e2da8e2f811d1448a5b465c236feacd80ffbac7b')   # same revision as Table 12
SIZES = (50, 100, 250, 500, 1000, None)            # None = all 6,420 training posts
SEEDS = (42, 43, 44)


def real_split(name):
    with (REAL / f'{name}.csv').open(newline='', encoding='utf-8') as handle:
        rows = [(r['tweet'], r['label']) for r in csv.DictReader(handle)]
    if {label for _, label in rows} != {'real', 'fake'}:
        raise ValueError(f'Unexpected labels in {name}.csv')
    return rows


def macro_f1_accuracy(true, pred):
    from sklearn.metrics import accuracy_score, f1_score
    return f1_score(true, pred, average='macro'), accuracy_score(true, pred)


def make_loader(tokenizer, texts, labels, batch_size, seed, shuffle):
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    encoded = tokenizer(list(texts), truncation=True, max_length=256)
    examples = [{**{k: v[i] for k, v in encoded.items()}, 'labels': int(labels[i])} for i in range(len(texts))]
    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(seed),
                      collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))


def predict(model, loader):
    import torch
    model.eval()
    out = []
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for batch in loader:
            out.extend(model(**{k: v.cuda() for k, v in batch.items() if k != 'labels'}).logits.argmax(-1).cpu().tolist())
    return out


def train(model, tokenizer, train_xy, val_xy, epochs, lr, batch_size, seed):
    """Fine-tune with AdamW; keep the epoch with the best validation Macro-F1."""
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    loader = make_loader(tokenizer, *train_xy, batch_size, seed, True)
    val_loader = make_loader(tokenizer, *val_xy, 64, seed, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    best, history = (-1.0, 0, None), []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model(**{k: v.cuda() for k, v in batch.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        score = macro_f1_accuracy(list(val_xy[1]), predict(model, val_loader))[0]
        history.append(score)
        if score > best[0]:
            best = (score, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    return best[1], best[0], history


def pretrain(root):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    target = root / OUT / 'hrb_roberta'
    if target.exists():
        raise FileExistsError(target)
    torch.use_deterministic_algorithms(True)
    records, provenance = load_corpus(root)
    train_rows, val_rows, _ = _train_validation(records, split_seed=42)
    labels = sorted({r.label for r in records})
    tokenizer = AutoTokenizer.from_pretrained(BASE[0], revision=BASE[1], token=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE[0], revision=BASE[1], token=False, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={l: i for i, l in enumerate(labels)}, attn_implementation='eager').cuda()
    xy = lambda rs: ([r.text for r in rs], [labels.index(r.label) for r in rs])
    epoch, val_f1, history = train(model, tokenizer, xy(train_rows), xy(val_rows), epochs=3, lr=1e-5, batch_size=32, seed=42)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    (target / 'hrb_training.json').write_text(json.dumps(dict(
        base=BASE, labels=labels, train_n=len(train_rows), validation_n=len(val_rows), best_epoch=epoch,
        validation_macro_f1=val_f1, history=history, corpus=provenance, runner_sha256=digest(Path(__file__))), indent=2))
    print('Saved HealthRoleBench RoBERTa', target, 'validation macro-F1', round(val_f1, 4), flush=True)


def transfer(root):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    source = root / OUT / 'hrb_roberta'
    target = root / OUT / 'transfer.json'
    if target.exists():
        raise FileExistsError(target)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source, attn_implementation='eager').cuda()
    test = real_split('test')
    pred = predict(model, make_loader(tokenizer, [t for t, _ in test], [0] * len(test), 64, 0, False))
    names = model.config.id2label
    mapped = ['real' if names[p] == 'Otherwise' else 'fake' for p in pred]
    f1, acc = macro_f1_accuracy([l for _, l in test], mapped)
    roles = {name: sum(names[p] == name for p in pred) for name in names.values()}
    target.write_text(json.dumps(dict(k=0, macro_f1=f1, accuracy=acc, predicted_roles=roles, test_predictions=mapped,
                                      mapping='any misinformation role -> fake; Otherwise -> real', n=len(test)), indent=2))
    print('Zero-shot transfer to real posts: macro-F1', round(f1, 4), 'accuracy', round(acc, 4), roles, flush=True)


def finetune(root, start, size, seed):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = f"{start}_k{size or 'all'}_seed{seed}"
    target = root / OUT / 'runs' / f'{name}.json'
    if target.exists():
        print('skip', name)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    train_rows, val_rows, test_rows = real_split('train'), real_split('val'), real_split('test')
    rng = random.Random(seed)
    if size:
        by_label = {l: [r for r in train_rows if r[1] == l] for l in ('fake', 'real')}
        sample = [r for l in ('fake', 'real') for r in rng.sample(by_label[l], size // 2)]
        rng.shuffle(sample)
    else:
        sample = train_rows
    label_ids = {'real': 0, 'fake': 1}
    xy = lambda rs: ([t for t, _ in rs], [label_ids[l] for _, l in rs])
    torch.manual_seed(seed)
    if start == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained(BASE[0], revision=BASE[1], token=False)
        model = AutoModelForSequenceClassification.from_pretrained(BASE[0], revision=BASE[1], token=False, num_labels=2,
                                                                   attn_implementation='eager')
    else:
        source = root / OUT / 'hrb_roberta'
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2, ignore_mismatched_sizes=True,
                                                                   id2label={0: 'real', 1: 'fake'}, label2id=label_ids,
                                                                   attn_implementation='eager')   # same as the RoBERTa arm
    model = model.cuda()
    epoch, val_f1, history = train(model, tokenizer, xy(sample), xy(val_rows), epochs=10, lr=2e-5, batch_size=16, seed=seed)
    pred = predict(model, make_loader(tokenizer, *xy(test_rows), 64, seed, False))
    f1, acc = macro_f1_accuracy(xy(test_rows)[1], pred)
    target.write_text(json.dumps(dict(start=start, k=len(sample), seed=seed, best_epoch=epoch, validation_macro_f1=val_f1,
                                      history=history, test_macro_f1=f1, test_accuracy=acc, test_predictions=pred,
                                      runner_sha256=digest(Path(__file__))), indent=2))
    print(name, 'test macro-F1', round(f1, 4), 'accuracy', round(acc, 4), 'epoch', epoch, flush=True)


def table(root):
    out = root / OUT
    runs = [json.loads(p.read_text()) for p in sorted((out / 'runs').glob('*.json'))]
    zero = json.loads((out / 'transfer.json').read_text())
    rows = [(0, 'zero-shot transfer', '', '', zero['macro_f1'], '', '', zero['accuracy'])]   # HealthRoleBench model only
    for k in sorted({r['k'] for r in runs}):
        cells = {}
        for start in ('roberta', 'hrb'):
            got = [r for r in runs if r['k'] == k and r['start'] == start]
            if len(got) == len(SEEDS):
                f1, acc = [r['test_macro_f1'] for r in got], [r['test_accuracy'] for r in got]
                cells[start] = (mean(f1), stdev(f1), mean(acc))
        if len(cells) == 2:
            rows.append((k, 'fine-tuned', *cells['roberta'][:2], *cells['hrb'][:2], cells['roberta'][2], cells['hrb'][2]))
    with (out / 'table_20_domain_adaptation.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('k_real_train', 'setting', 'roberta_macro_f1', 'roberta_sd', 'hrb_macro_f1', 'hrb_sd',
                         'roberta_accuracy', 'hrb_accuracy'))
        writer.writerows(rows)
    (out / 'manifest.json').write_text(json.dumps(dict(
        real_data=json.loads((root / REAL / 'SOURCE.json').read_text()),
        real_sha256={p.name: digest(p) for p in sorted((root / REAL).glob('*.csv'))},
        base=BASE, sizes=[s or 'all' for s in SIZES], seeds=SEEDS, runs=len(runs), runner_sha256=digest(Path(__file__))), indent=2))
    print('Table 20 written with', len(runs), 'fine-tuning runs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Table 20: domain adaptation to real COVID-19 posts.')
    parser.add_argument('step', choices=('pretrain', 'transfer', 'finetune', 'table', 'all'))
    parser.add_argument('--project-root', required=True, type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.step in ('pretrain', 'all') and not (root / OUT / 'hrb_roberta').exists():
        pretrain(root)
    if args.step in ('transfer', 'all') and not (root / OUT / 'transfer.json').exists():
        transfer(root)
    if args.step in ('finetune', 'all'):
        for size in SIZES:
            for start in ('roberta', 'hrb'):
                for seed in SEEDS:
                    finetune(root, start, size, seed)
    if args.step in ('table', 'all'):
        table(root)
