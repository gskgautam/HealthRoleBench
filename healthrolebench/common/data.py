import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

MODELS = ('gpt4o', 'qwen', 'llama2', 'mistral', 'gemma')
DISEASES = ('COVID-19', 'HPV', 'Influenza')


@dataclass(frozen=True)
class Record:
    generator: str
    sample_id: str
    disease: str
    label: str
    text: str
    group: int


def digest(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        yield from csv.DictReader(handle)


RELEASE_CSV = Path('data') / 'healthrolebench_v1.csv'
RELEASE_SHA256 = '45037b80a64194a608aec2fb08a883a45e5644b42686be64f3ef153395f0bcaf'


def load_release(root):
    """Load the public corpus (posts, labels, leakage groups) released with the paper."""
    path = root / RELEASE_CSV
    if digest(path) != RELEASE_SHA256:
        raise ValueError(f'Unexpected corpus file: {path}')
    records = [Record(r['generator'], r['sample_id'], r['disease'], r['label'], r['text'], int(r['group'])) for r in rows(path)]
    if len(records) != 60_000:
        raise ValueError('Release corpus must contain 60,000 posts')
    return records, dict(release_sha256=RELEASE_SHA256)


def load_corpus(root):
    if (root / RELEASE_CSV).exists():
        return load_release(root)
    folder = root / 'data' / 'experiments'
    manifest_path = folder / 'corpus_tables_v1' / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    group_path = folder / 'full_transfer_pilot_v1' / 'groups.csv'
    group_rows = list(rows(group_path))
    groups = {(row['generator'], row['sample_id']): row for row in group_rows}
    if len(groups) != len(group_rows):
        raise ValueError('Repeated group-manifest IDs')
    records = []
    source_hashes = {}
    for model in MODELS:
        paths = {
            'bulk': Path('data/filter_runs/curation_v2') / model / 'selected_machine_curated_draft.csv',
            'topup': Path('data/topups_v2') / f'{model}_candidates.csv',
        }
        for stage, relative in paths.items():
            source = root / relative
            actual = digest(source)
            if actual != manifest['input_sha256'][str(relative)]:
                raise ValueError(f'Corpus input changed: {relative}')
            source_hashes[str(relative)] = actual
            for row in rows(source):
                key = model, row['sample_id']
                group = groups.pop(key)
                if (group['disease'], group['role'], group['stage']) != (row['disease'], row['intended_role'], stage):
                    raise ValueError(f'Group metadata mismatch: {key}')
                records.append(Record(*key, row['disease'], row['intended_role'], row['text'], int(group['group'])))
    if groups or len(records) != 60_000:
        raise ValueError('Group manifest does not cover all selected records')
    return records, dict(group_sha256=digest(group_path), input_sha256=source_hashes)


def _train_validation(available, split_seed):
    """Group-disjoint train/validation split (~20% validation per class) of the available records."""
    labels = [r.label for r in available]
    expected = Counter(labels)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    splits = []
    for train_idx, val_idx in splitter.split(np.zeros(len(available)), labels, [r.group for r in available]):
        train = [available[i] for i in train_idx]
        val = [available[i] for i in val_idx]
        train_labels, val_labels = Counter(r.label for r in train), Counter(r.label for r in val)
        if train_labels.keys() != expected.keys() or val_labels.keys() != expected.keys():
            continue
        deviation = sum(abs(val_labels[label] / count - .2) for label, count in expected.items())
        splits.append((deviation, train, val))
    if not splits:
        raise ValueError('No group-separated split with all classes in train and validation')
    _, train, val = min(splits, key=lambda result: result[0])
    return train, val, expected


def _check(train, val, test, expected):
    group_sets = [{r.group for r in partition} for partition in (train, val, test)]
    if any(group_sets[a] & group_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Related records cross split boundaries')
    if set(r.label for r in test) != expected.keys():
        raise ValueError('Class mismatch between training and test data')


def split_disease(records, held_out, split_seed):
    """Train on the two other diseases, test on `held_out` (Table 12; Tables 7, 13, 14)."""
    test = [r for r in records if r.disease == held_out]
    test_groups = {r.group for r in test}
    candidates = [r for r in records if r.disease != held_out]
    available = [r for r in candidates if r.group not in test_groups]
    if not test or not available:
        raise ValueError('Held-out disease or training data is empty')
    train, val, expected = _train_validation(available, split_seed)
    _check(train, val, test, expected)
    return train, val, test, len(candidates) - len(available)


def split_source(records, source, split_seed):
    """Train on `source` only, test on each of the two other diseases (Table 11)."""
    test = [r for r in records if r.disease != source]
    test_groups = {r.group for r in test}
    candidates = [r for r in records if r.disease == source]
    available = [r for r in candidates if r.group not in test_groups]
    if not test or not available:
        raise ValueError('Source disease or test data is empty')
    train, val, expected = _train_validation(available, split_seed)
    _check(train, val, test, expected)
    return train, val, test, len(candidates) - len(available)
