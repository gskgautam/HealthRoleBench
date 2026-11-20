"""Freeze the balanced held-out evaluation set used by Tables 8, 9, 10, 13 (overlap) and 18.

Design (fixed before any model is scored on it):
- 20 posts per generator x disease x role cell (5 x 3 x 5 x 20 = 1,500), so every
  generator contributes 300 posts and every role is equally represented.
- No two selected posts are near-duplicates (binary word-trigram cosine >= 0.80, the
  threshold used for leakage grouping), and none is a near-duplicate of an audit post.
  Exemplar-linked groups are deliberately not used here: they chain whole cells together
  (REVIEW.md A3), which would make some cells unfillable. Leakage between this set and
  training data is handled per experiment (see probe_full.py).
- Disjoint from the 450-post human-audit sample, which stays reserved for Tables 3 and 17.
- Deterministic: posts are ranked by SHA-256 of a fixed salt, generator and sample ID.

Run from PROJECT_ROOT:  python -m healthrolebench.curation.eval_set --project-root .
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from healthrolebench.common.artifacts import private_text_artifact
from healthrolebench.common.data import DISEASES, MODELS, digest, load_corpus, rows

PER_CELL = 20
SALT = 'healthrolebench-eval-v1|42'
SIMILARITY = .80
OUT = Path('data/experiments/eval_set_v1')


def rank(record):
    return hashlib.sha256(f'{SALT}|{record.generator}|{record.sample_id}'.encode()).hexdigest()


def build(root):
    out = root / OUT
    if out.exists():
        raise FileExistsError(out)
    records, provenance = load_corpus(root)
    audit_path = root / 'data/experiments/corpus_tables_v1/audit_key_private.csv'
    audit = {(r['generator'], r['sample_id']) for r in rows(audit_path)}
    by_key = {(r.generator, r.sample_id): r for r in records}
    labels = sorted({r.label for r in records})
    cells = defaultdict(list)
    for record in records:
        if (record.generator, record.sample_id) not in audit:
            cells[record.generator, record.disease, record.label].append(record)
    vectorizer = CountVectorizer(ngram_range=(3, 3), binary=True, dtype=np.float32).fit(r.text for r in records)
    embed = lambda texts: normalize(vectorizer.transform(texts))
    kept = embed([by_key[k].text for k in sorted(audit)])     # audit posts block their near-duplicates
    chosen = []
    for key in sorted(cells):
        picked = []
        for record in sorted(cells[key], key=rank):
            vector = embed([record.text])
            if kept.shape[0] and (kept @ vector.T).max() >= SIMILARITY:
                continue
            picked.append(record)
            kept = vstack([kept, vector]).tocsr()
            if len(picked) == PER_CELL:
                break
        if len(picked) != PER_CELL:
            raise ValueError(f'Cell {key} has only {len(picked)} eligible posts')
        chosen.extend(picked)
    expected = len(MODELS) * len(DISEASES) * len(labels) * PER_CELL
    if len(chosen) != expected:
        raise ValueError(f'Expected {expected} posts, got {len(chosen)}')
    order = sorted(chosen, key=rank)          # shuffle away from cell order
    out.mkdir(parents=True)
    items = [(f'EVAL-{i:04d}', r) for i, r in enumerate(order, 1)]
    with private_text_artifact(out / 'eval_key_private.csv') as handle:
        writer = csv.writer(handle)
        writer.writerow(('item_id', 'generator', 'sample_id', 'disease', 'intended_role', 'group'))
        writer.writerows((item, r.generator, r.sample_id, r.disease, r.label, r.group) for item, r in items)
    with private_text_artifact(out / 'eval_posts_private.csv') as handle:
        writer = csv.writer(handle)
        writer.writerow(('item_id', 'post'))
        writer.writerows((item, r.text) for item, r in items)
    manifest = dict(per_cell=PER_CELL, n=len(items), salt=SALT, labels=labels, generators=list(MODELS),
                    diseases=list(DISEASES), near_duplicate_cosine=SIMILARITY, excludes_audit_posts=len(audit),
                    audit_key_sha256=digest(audit_path), corpus=provenance,
                    label_status='machine-curated intended roles')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + chr(10), encoding='utf-8')
    print('Evaluation set frozen:', out, len(items), 'posts')


def load_eval(root):
    """Return (key rows by item_id, post text by item_id)."""
    folder = root / OUT
    key = {r['item_id']: r for r in rows(folder / 'eval_key_private.csv')}
    posts = {r['item_id']: r['post'] for r in rows(folder / 'eval_posts_private.csv')}
    if key.keys() != posts.keys() or not key:
        raise ValueError('Evaluation key and posts differ')
    return key, posts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--project-root', required=True, type=Path)
    build(parser.parse_args().project_root.resolve())
