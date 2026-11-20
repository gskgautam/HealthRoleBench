"""Table 17: single-label vs multi-label vs soft-label classification on the human-audited posts.

Needs the completed audit (annotations/healthrolebench_audit_v1/annotator_a.csv, annotator_b.csv,
adjudicated.csv; see human_audit.py). Posts adjudicated UNSURE are excluded.

Targets
  hard multi-label: adjudicated primary role plus adjudicated secondary roles
  soft labels:      share of the two annotators naming each role (primary or secondary)
Settings (5-fold cross-validation over the resolved posts, fewer folds only if a role has <5 posts; stratified by primary role, seed 42;
features: TF-IDF word 1-2 grams fitted on training folds only; logistic regression, C=1)
  1 single-label (original)         multinomial classifier on the primary role
  2 multi-label (binary relevance)  one binary classifier per role, threshold 0.5
  3 multi-label (threshold-tuned)   as 2, per-role threshold maximising F1 on an inner 3-fold split
  4 soft-label                      per-role classifier fitted to soft targets (weighted positive/negative copies)
Predictions are written in the audit-file format and scored by human_audit.score, which produces
table_17_multilabel_evaluation.csv (Macro-F1, Micro-F1, Hamming loss, samples-Jaccard).

Run from PROJECT_ROOT:  python -m healthrolebench.analysis.multilabel --project-root .
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from healthrolebench.common.artifacts import private_text_artifact
from healthrolebench.common.data import rows
from healthrolebench.curation.human_audit import ROLES, UNSURE, index, labels_from_row, score

AUDIT = Path('annotations/healthrolebench_audit_v1')
OUT = Path('data/experiments/multilabel_v1')
THRESHOLDS = np.round(np.arange(.1, .91, .05), 2)
SETTINGS = ('Single-label (original)', 'Multi-label (binary relevance)',
            'Multi-label (threshold-tuned)', 'Soft-label (annotator distribution)')


def to_labels(primary_scores, multi, thresholds):
    """Map role scores to (primary, secondary) under the audit rules (Otherwise is exclusive)."""
    primary = ROLES[int(np.argmax(primary_scores))]
    if primary == 'Otherwise':
        return primary, ()
    secondary = tuple(r for r, s, t in zip(ROLES, multi, thresholds) if r not in (primary, 'Otherwise') and s >= t)
    return primary, secondary


def fit_binary(x, y, weight=None):
    if len(set(y)) < 2:                       # constant role in this fold
        constant = float(y[0])
        return lambda z: np.full(z.shape[0], constant)
    model = LogisticRegression(C=1.0, max_iter=2000).fit(x, y, sample_weight=weight)
    return lambda z: model.predict_proba(z)[:, 1]


def tune_threshold(texts, y, seed):
    folds = StratifiedKFold(3, shuffle=True, random_state=seed) if min(np.bincount(y, minlength=2)) >= 3 else None
    if folds is None:
        return .5
    scores = np.zeros(len(y))
    for a, b in folds.split(texts, y):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(texts[a])
        scores[b] = fit_binary(vec.transform(texts[a]), y[a])(vec.transform(texts[b]))
    def f1(t):
        p = scores >= t
        tp = (p & (y == 1)).sum()
        return 2 * tp / (p.sum() + y.sum()) if tp else 0.0
    return float(max(THRESHOLDS, key=f1))


def predict(texts, hard, soft, primary, seed=42):
    n = len(texts)
    outputs = {s: [None] * n for s in SETTINGS}
    smallest = min(np.unique(primary, return_counts=True)[1])
    if smallest < 2:
        raise SystemExit('Every adjudicated primary role needs at least 2 posts for cross-validation')
    folds = StratifiedKFold(min(5, smallest), shuffle=True, random_state=seed)   # 5 folds unless a role is rarer
    for train, test in folds.split(texts, primary):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(texts[train])
        xtr, xte = vec.transform(texts[train]), vec.transform(texts[test])
        single = LogisticRegression(C=1.0, max_iter=2000).fit(xtr, primary[train])
        proba = np.zeros((len(test), len(ROLES)))
        proba[:, [ROLES.index(c) for c in single.classes_]] = single.predict_proba(xte)
        binary = np.column_stack([fit_binary(xtr, hard[train, j])(xte) for j in range(len(ROLES))])
        tuned = [tune_threshold(texts[train], hard[train, j], seed) for j in range(len(ROLES))]
        soft_scores = []
        for j in range(len(ROLES)):
            target = soft[train, j]
            x2 = vstack([xtr, xtr]).tocsr()
            y2 = np.r_[np.ones(len(train)), np.zeros(len(train))]
            w2 = np.r_[target, 1 - target]
            keep = w2 > 0
            soft_scores.append(fit_binary(x2[keep], y2[keep], w2[keep])(xte) if len(set(y2[keep])) > 1
                               else np.full(len(test), float(target.mean())))
        soft_scores = np.column_stack(soft_scores)
        for k, i in enumerate(test):
            outputs[SETTINGS[0]][i] = (ROLES[int(np.argmax(proba[k]))], ())
            outputs[SETTINGS[1]][i] = to_labels(binary[k], binary[k], [.5] * len(ROLES))
            outputs[SETTINGS[2]][i] = to_labels(binary[k], binary[k], tuned)
            outputs[SETTINGS[3]][i] = to_labels(soft_scores[k], soft_scores[k], [.5] * len(ROLES))
    return outputs


def main(root):
    audit = root / AUDIT
    paths = [audit / f for f in ('annotator_a.csv', 'annotator_b.csv', 'adjudicated.csv')]
    if not all(p.is_file() for p in paths):
        raise SystemExit('Table 17 needs the completed audit files: ' + ', '.join(map(str, paths)))
    posts = {r['audit_id']: r['post'] for r in rows(root / 'data/experiments/corpus_tables_v1/audit_blinded.csv')}
    a, b, final = ({k: labels_from_row(v) for k, v in index(p).items()} for p in paths)
    ids = sorted(k for k, (primary, _) in final.items() if primary != UNSURE)
    if len(ids) < 50:
        raise SystemExit(f'Only {len(ids)} resolved posts; at least 50 are needed for 5-fold cross-validation')
    texts = np.array([posts[k] for k in ids], dtype=object)
    hard = np.array([[int(r == final[k][0] or r in final[k][1]) for r in ROLES] for k in ids])
    soft = np.array([[sum(r == p or r in s for p, s in (a[k], b[k])) / 2 for r in ROLES] for k in ids], dtype=float)
    primary = np.array([final[k][0] for k in ids])
    outputs = predict(texts, hard, soft, primary)
    out = root / OUT
    out.mkdir(parents=True, exist_ok=False)
    files = []
    for n, setting in enumerate(SETTINGS, 1):
        path = out / f'setting_{n}.csv'
        with private_text_artifact(path) as handle:
            writer = csv.writer(handle)
            writer.writerow(('audit_id', 'primary_role', 'secondary_roles'))
            writer.writerows((k, p, '|'.join(s)) for k, (p, s) in zip(ids, outputs[setting]))
        files.append((setting, str(path)))
    score(root, *paths, files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Table 17 model settings on the human-audited posts.')
    parser.add_argument('--project-root', required=True, type=Path)
    main(parser.parse_args().project_root.resolve())
