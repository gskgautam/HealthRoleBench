import argparse
import csv
import json
import os
import random
import socket
import tempfile
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import sklearn
import torch
import transformers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from healthrolebench.common.data import DISEASES, digest, load_corpus, split_disease, split_source

MODELS = {
    'bert': ('google-bert/bert-base-uncased', '86b5e0934494bd15c9632b12f734a8a67f723594'),
    'roberta': ('FacebookAI/roberta-base', 'e2da8e2f811d1448a5b465c236feacd80ffbac7b'),
    'distilbert': ('distilbert/distilbert-base-uncased', '12040accade4e8a0f71eabdb258fecc2e7e948be'),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train a standard encoder: two diseases -> third (--disease, Table 12) '
                                     'or one disease -> each other disease (--source-disease, Table 11).')
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--model', choices=MODELS, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--disease', choices=DISEASES, help='Held-out test disease')
    target.add_argument('--source-disease', choices=DISEASES, help='Single training disease')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()


def predict(model, loader):
    model.eval()
    predictions = []
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for batch in loader:
            inputs = {key: val.to('cuda:0') for key, val in batch.items() if key != 'labels'}
            predictions.extend(model(**inputs).logits.argmax(dim=-1).cpu().tolist())
    return predictions


def make_loader(records, tokenizer, label_ids, batch_size, seed, shuffle):
    tokenized = tokenizer([row.text for row in records], truncation=True, max_length=256)
    examples = [{**{key: values[i] for key, values in tokenized.items()}, 'labels': label_ids[row.label]}
                for i, row in enumerate(records)]
    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(seed),
                      collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8), num_workers=0)


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU with bfloat16 support is required')
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError('Epochs and batch size must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model_id, revision = MODELS[args.model]
    root = args.project_root.resolve()
    protocol, domain = ('held_out', args.disease) if args.disease else ('single_source', args.source_disease)
    folder = 'encoder_transfer_v1' if protocol == 'held_out' else 'encoder_source_v1'
    output = root / 'data' / 'experiments' / folder / args.model / domain / f'seed_{args.seed}'
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records, provenance = load_corpus(root)
    split = split_disease if protocol == 'held_out' else split_source
    train, validation, test, excluded = split(records, domain, split_seed=42)
    classes = sorted({r.label for r in train})
    label_ids = {name: i for i, name in enumerate(classes)}
    print('loading', model_id, 'train/validation/test:', len(train), len(validation), len(test), 'purged:', excluded, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, token=False, use_safetensors=True, num_labels=len(classes),
        id2label=dict(enumerate(classes)), label2id=label_ids, attn_implementation='eager').to('cuda:0')
    loaders = [make_loader(rows, tokenizer, label_ids, args.batch_size, args.seed, i == 0)
               for i, rows in enumerate((train, validation, test))]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.01)
    validation_true = [label_ids[r.label] for r in validation]
    best_f1, best_epoch, best_state = -1.0, 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for step, batch in enumerate(loaders[0], 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model(**{key: val.to('cuda:0') for key, val in batch.items()}).loss
            if not loss.isfinite().item():
                raise FloatingPointError(f'Non-finite loss at epoch {epoch}, step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item()
            if step == 1 or step % 200 == 0:
                print('epoch', epoch, 'step', step, '/', len(loaders[0]), 'loss', round(loss_sum/step, 5), flush=True)
        guessed = predict(model, loaders[1])
        score = f1_score(validation_true, guessed, labels=range(len(classes)), average='macro', zero_division=0)
        history.append(dict(epoch=epoch, validation_macro_f1=score, mean_step_train_loss=loss_sum/len(loaders[0])))
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            best_state = {name: val.detach().cpu().clone() for name, val in model.state_dict().items()}
        print('validation', history[-1], flush=True)
    model.load_state_dict(best_state)
    prediction = [classes[i] for i in predict(model, loaders[2])]
    true = [r.label for r in test]
    matrix = confusion_matrix(true, prediction, labels=classes)
    if matrix.sum() != len(test):
        raise ValueError('Confusion matrix does not cover all held-out records')
    per_target = {}
    if protocol == 'single_source':
        for disease in sorted({r.disease for r in test}):
            idx = [i for i, r in enumerate(test) if r.disease == disease]
            t, p = [true[i] for i in idx], [prediction[i] for i in idx]
            per_target[disease] = dict(n=len(idx), accuracy=accuracy_score(t, p),
                                       classification_report=classification_report(t, p, labels=classes, output_dict=True, zero_division=0),
                                       confusion_matrix=confusion_matrix(t, p, labels=classes).tolist())
    result = dict(model=model_id, revision=revision, protocol=protocol, disease=domain, train_seed=args.seed, split_seed=42,
                  batch_size=args.batch_size, epochs=args.epochs, max_tokens=256, learning_rate=1e-5, weight_decay=.01,
                  best_epoch=best_epoch, training_history=history, train_n=len(train), validation_n=len(validation),
                  test_n=len(test), purged_train_n=excluded, labels=classes, label_status='machine-curated intended labels',
                  accuracy=accuracy_score(true,prediction),
                  classification_report=classification_report(true,prediction,labels=classes,output_dict=True,zero_division=0),
                  confusion_matrix=matrix.tolist(), per_target=per_target, provenance=provenance, host=socket.gethostname(),
                  gpu=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  versions={'torch':torch.__version__,'transformers':transformers.__version__,'sklearn':sklearn.__version__},
                  runner_sha256=digest(Path(__file__)))
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.pending-') as tmp:
        temporary = Path(tmp) / 'result'
        temporary.mkdir()
        with (temporary/'predictions.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('generator','sample_id','label','prediction') + (('disease',) if per_target else ()))
            writer.writerows((r.generator,r.sample_id,r.label,p) + ((r.disease,) if per_target else ()) for r,p in zip(test,prediction))
        with (temporary/'split.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('partition','generator','sample_id','group'))
            writer.writerows((partition,r.generator,r.sample_id,r.group) for partition,rows in
                             (('train',train),('validation',validation),('test',test)) for r in rows)
        (temporary/'metrics.json').write_text(json.dumps(result,indent=2)+chr(10),encoding='utf-8')
        temporary.rename(output)
    print('Completed:', output, 'macro-F1:', result['classification_report']['macro avg']['f1-score'], flush=True)


if __name__ == '__main__':
    run(parse_args())
