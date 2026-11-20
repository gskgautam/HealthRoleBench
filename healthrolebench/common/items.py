"""Inputs for zero-shot classification: which posts, under which condition.

items='audit'  the frozen 450-post human-audit sample (pilot runs, Tables 3/17 pairing)
items='eval'   the frozen 1,500-post evaluation set (Tables 8, 9, 10, 13, 18; Figure 8)

Conditions (Table 18):
  original      the post as curated
  cue-masked    role-lexicon terms replaced by [MASKED]
  paraphrased   meaning-preserving rewrite        (eval only; see perturb.py)
  adversarial   rewrite that hides overt cues     (eval only; see perturb.py)
Rewrites are used only where the curation NLI judge still assigns the intended role.
"""
import csv
from pathlib import Path

CONDITIONS = ('original', 'cue-masked', 'paraphrased', 'adversarial')
EXPERIMENTS = Path('data/experiments')


def _read(path):
    with path.open(newline='', encoding='utf-8') as source:
        return list(csv.DictReader(source))


def default_output(items, condition):
    if items == 'audit':
        return 'zero_shot_audit_v2' if condition == 'original' else 'zero_shot_cue_mask_v2'
    return f'zero_shot_eval_v1/{condition}'


def load_items(items, condition, root=Path('.')):
    """Return (id_column, [(item_id, post), ...]) for the requested sample and condition."""
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown condition {condition!r}')
    if items == 'audit':
        if condition in ('paraphrased', 'adversarial'):
            raise ValueError('Rewritten conditions exist only for the evaluation set')
        id_column = 'audit_id'
        data = [(r['audit_id'], r['post']) for r in _read(root / EXPERIMENTS / 'corpus_tables_v1/audit_blinded.csv')]
        expected = 450
    elif items == 'eval':
        id_column = 'item_id'
        folder = root / EXPERIMENTS / 'eval_set_v1'
        if condition in ('paraphrased', 'adversarial'):
            data = [(r['item_id'], r['rewrite']) for r in _read(folder / f'perturbed_{condition}.csv') if r['valid'] == '1']
            expected = None                     # only label-preserving rewrites are scored
        else:
            data = [(r['item_id'], r['post']) for r in _read(folder / 'eval_posts_private.csv')]
            expected = 1500
    else:
        raise ValueError(f'Unknown item set {items!r}')
    ids = [item for item, _ in data]
    if len(set(ids)) != len(ids) or '' in ids or (expected and len(ids) != expected):
        raise ValueError(f'Unexpected item IDs for {items}/{condition}: {len(ids)}')
    if condition == 'cue-masked':
        from healthrolebench.reporting.stress_tables import masker
        pattern = masker(root / 'configs/filter_v1/protocol.json')
        data = [(item, pattern.sub('[MASKED]', post)) for item, post in data]
    return id_column, data
