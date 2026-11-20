import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

def main():
    root = Path('data')
    out = root / 'experiments' / 'corpus_tables_v1'
    links = []
    counts = Counter()
    for model in ('gpt4o', 'qwen', 'llama2', 'mistral', 'gemma'):
        base = root/'filter_runs'/'curation_v2'/model/'selected_machine_curated_draft.csv'
        topup = root/'topups_v2'/f'{model}_candidates.csv'
        manifest = json.loads((root/'topups_v2'/f'{model}_manifest.json').read_text())
        assert hashlib.sha256(base.read_bytes()).hexdigest() == manifest['seed_selection_sha256'], model
        cells, roles = defaultdict(list), defaultdict(list)
        with base.open(newline='', encoding='utf-8') as source:
            for row in csv.DictReader(source):
                cells[row['disease'], row['intended_role']].append(row)
                roles[row['intended_role']].append(row)
        with topup.open(newline='', encoding='utf-8') as source:
            for row in csv.DictReader(source):
                disease, role = row['disease'], row['intended_role']
                pool = cells[disease, role] or roles[role]
                examples = random.Random(int(row['seed'])).sample(pool, min(2, len(pool)))
                counts['topups'] += 1
                counts['without_exemplars'] += not examples
                counts['cross_disease_topups'] += any(example['disease'] != disease for example in examples)
                prompt = row['prompt']
                for example in examples:
                    text = example['text']
                    if text not in prompt and json.dumps(text, ensure_ascii=False)[1:-1] not in prompt:
                        counts['unverified_links'] += 1
                        continue
                    links.append((model, row['sample_id'], disease, example['sample_id'], example['disease']))
    assert counts['topups'] == 16641
    with (out/'topup_exemplar_links.csv').open('x', newline='', encoding='utf-8') as target:
        writer = csv.writer(target)
        writer.writerow(('generator','topup_id','topup_disease','exemplar_id','exemplar_disease'))
        writer.writerows(links)
    print(dict(counts), 'verified_links', len(links))


if __name__ == '__main__':
    main()
