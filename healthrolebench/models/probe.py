import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from healthrolebench.common.artifacts import private_text_artifact
from healthrolebench.common.data import digest, load_corpus, rows
from healthrolebench.models.zero_shot import CACHE, MODELS

SEEDS = (42,43,44)
CAP = 60


def design(root):
    records, source = load_corpus(root)
    path = root/'data/experiments/corpus_tables_v1/audit_key_private.csv'
    key = list(rows(path))
    by_id = {(r.generator,r.sample_id): r for r in records}
    test = [by_id[r['generator'],r['sample_id']] for r in key]
    if len(test) != 450 or len({r['audit_id'] for r in key}) != len(test):
        raise ValueError('Unexpected audit IDs')
    if any(record.label != row['intended_role'] for record,row in zip(test,key)):
        raise ValueError('Audit labels do not match the selected corpus')
    held_groups = {r.group for r in test}
    buckets = defaultdict(list)
    for record in records:
        if record.group not in held_groups:
            buckets[record.generator,record.label].append(record)
    rng = random.Random(42)
    train = [r for _,items in sorted(buckets.items()) for r in rng.sample(items,min(CAP,len(items)))]
    labels = {r.label for r in test}
    if len(labels) != 5 or any({r.label for r in train if r.generator != generator} != labels
                              for generator in {r.generator for r in test}):
        raise ValueError('A generator holdout has no training support for a role')
    source['training_bucket_sizes'] = {generator: {label: min(CAP,len(buckets[generator,label]))
                                                    for label in sorted(labels)}
                                       for generator in sorted({r.generator for r in test})}
    source['audit_key_sha256'] = digest(path)
    return train,test,key,source


def train_probe(model, root):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.use_deterministic_algorithms(True)
    train,test,key,source = design(root)
    folder, revision = MODELS[model]
    cache = root/'models' if model == 'gemma' else CACHE
    snapshot = cache/('models--'+folder)/'snapshots'/revision
    tokenizer = AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(snapshot,local_files_only=True)
    model_class = AutoModelForImageTextToText if config.model_type == 'qwen3_5' else AutoModelForCausalLM
    backbone = model_class.from_pretrained(snapshot,local_files_only=True,dtype=torch.float16,attn_implementation='eager').to('cuda:0').eval()
    base = backbone.model.language_model if config.model_type == 'qwen3_5' else backbone.model
    chunks = []
    print(model, 'embedding', len(train), 'training and', len(test), 'test samples', flush=True)
    with torch.inference_mode():
        records = train + test
        for start in range(0,len(records),12):
            batch = tokenizer([r.text for r in records[start:start+12]], padding=True, truncation=True,
                              max_length=256, return_tensors='pt').to('cuda:0')
            states = base(**batch,use_cache=False,return_dict=True).last_hidden_state.float()
            attention = batch.attention_mask.unsqueeze(-1)
            chunks.append(((states*attention).sum(dim=1)/attention.sum(dim=1).clamp_min(1)).cpu())
            if start % 600 == 0:
                print(model, 'embeddings', min(start+12,len(records)), '/', len(records), flush=True)
    del base, backbone
    torch.cuda.empty_cache()
    features = torch.nn.functional.normalize(torch.cat(chunks).to('cuda:0'), dim=1)
    labels = sorted({r.label for r in train})
    targets = torch.tensor([labels.index(r.label) for r in train],device='cuda:0')
    out = root/'data/experiments/frozen_head_audit_v1'
    out.mkdir(parents=True,exist_ok=True)
    training = out/f'{model}_training_ids.csv'
    if not training.exists():
        with private_text_artifact(training) as stream:
            writer = csv.writer(stream)
            writer.writerow(('generator','sample_id','role','group'))
            writer.writerows((r.generator,r.sample_id,r.label,r.group) for r in train)
    results = []
    for generator in sorted({r.generator for r in test}):
        training_ids = [i for i,r in enumerate(train) if r.generator != generator]
        test_ids = [i for i,r in enumerate(test) if r.generator == generator]
        if len(set(targets[training_ids].tolist())) != len(labels):
            raise ValueError(f'Training labels incomplete for {generator} holdout')
        for seed in SEEDS:
            torch.manual_seed(seed)
            head = torch.nn.Linear(features.shape[1],len(labels),device='cuda:0')
            optimizer = torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.01)
            train_x, train_y = features[training_ids], targets[training_ids]
            for _ in range(80):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(head(train_x),train_y)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                prediction = head(features[[len(train)+i for i in test_ids]]).argmax(dim=1).tolist()
            results.extend((seed,key[i]['audit_id'],generator,labels[p]) for i,p in zip(test_ids,prediction))
    if len(results) != len(test)*len(SEEDS):
        raise ValueError('Incomplete frozen-head predictions')
    with private_text_artifact(out/f'{model}_manifest.json') as stream:
        json.dump(dict(inputs=source, revision=revision, train_n=len(train), test_n=len(test), seeds=SEEDS,
                       epochs=80, optimizer='AdamW', lr=.01, max_tokens=256, classifier='frozen mean-pooled backbone + linear head',
                       label_status='machine-curated intended roles', gpu=torch.cuda.get_device_name(0),
                       torch=torch.__version__, cuda=torch.version.cuda, runner_sha256=digest(Path(__file__))), stream, indent=2)
    with private_text_artifact(out/f'{model}.csv') as stream:
        writer = csv.writer(stream)
        writer.writerow(('seed','audit_id','held_generator','prediction'))
        writer.writerows(results)
    print('Completed', model, len(results), 'frozen-head audit predictions', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit a frozen-backbone classifier head for generator-held-out audit examples.')
    parser.add_argument('--project-root',required=True,type=Path)
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--check-design',action='store_true')
    args = parser.parse_args()
    if args.check_design:
        train,test,_,info = design(args.project_root.resolve())
        print('Disjoint probe design:',len(train),'training and',len(test),'audit samples')
        print('Empty buckets:',sum(n == 0 for counts in info['training_bucket_sizes'].values() for n in counts.values()))
    elif args.model:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        train_probe(args.model,args.project_root.resolve())
    else:
        parser.error('Provide --model or --check-design')
