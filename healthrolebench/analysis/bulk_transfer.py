import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize

MODELS = ('gpt4o', 'qwen', 'llama2', 'mistral', 'gemma')
NEAR_DUPLICATE_COSINE = 0.80
SEED = 42


def run(output):
    if output.exists():
        raise FileExistsError(output)
    rows = []
    input_hashes = {}
    for model in MODELS:
        path = Path('data/filter_runs/curation_v2') / model / 'selected_machine_curated_draft.csv'
        input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline='', encoding='utf-8') as source:
            rows.extend((model, row['sample_id'], row['disease'], row['intended_role'], row['text'])
                        for row in csv.DictReader(source))
    assert len(rows) == 43359 and len({r[:2] for r in rows}) == len(rows)
    generator, sample_id, disease, role, text = np.asarray(rows, dtype=object).T
    labels = sorted(set(role))
    assert len(labels) == 5 and len(set(disease)) == 3
    # Binary word trigrams for similarity screening only. Fit classifier features on training data per fold.
    trigrams = normalize(CountVectorizer(ngram_range=(3, 3), binary=True, dtype=np.float32).fit_transform(text))
    output.mkdir(parents=True)
    metrics = []
    with (output / 'predictions.csv').open('x', newline='', encoding='utf-8') as target:
        writer = csv.writer(target)
        writer.writerow(('protocol', 'held_out', 'generator', 'sample_id', 'disease', 'intended_role', 'prediction'))
        for protocol, domains in (('generator', generator), ('disease', disease)):
            for held_out in sorted(set(domains)):
                train = np.flatnonzero(domains != held_out)
                test = np.flatnonzero(domains == held_out)
                reference = trigrams[test].T.tocsr()
                leaky = np.zeros(train.size, dtype=bool)
                for start in range(0, train.size, 128):
                    block = trigrams[train[start:start+128]] @ reference
                    leaky[start:start+128] = np.asarray((block >= NEAR_DUPLICATE_COSINE).getnnz(axis=1)).ravel() > 0
                train = train[~leaky]
                assert set(role[train]) == set(role[test]) == set(labels)
                classifier = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                           LinearSVC(class_weight='balanced', random_state=SEED))
                classifier.fit(text[train], role[train])
                prediction = classifier.predict(text[test])
                report = classification_report(role[test], prediction, labels=labels, output_dict=True, zero_division=0)
                matrix = confusion_matrix(role[test], prediction, labels=labels)
                assert matrix.sum() == test.size and matrix.sum(axis=0).sum() == matrix.sum(axis=1).sum()
                metrics.append(dict(protocol=protocol, held_out=held_out, train_n=int(train.size), test_n=int(test.size),
                                    near_duplicate_train_exclusions=int(leaky.sum()), accuracy=accuracy_score(role[test], prediction),
                                    report=report, confusion_matrix=matrix.tolist()))
                writer.writerows((protocol, held_out, generator[j], sample_id[j], disease[j], role[j], pred)
                                 for j, pred in zip(test, prediction))
                print(f'{protocol} {held_out}: train={train.size} test={test.size} excluded={leaky.sum()} macro-F1={report["macro avg"]["f1-score"]:.4f}', flush=True)
    for protocol in ('generator', 'disease'):
        assert sum(m['test_n'] for m in metrics if m['protocol'] == protocol) == len(rows)
    result = dict(stage='bulk only', gold='machine-curated intended labels', labels=labels,
                  seed=SEED, near_duplicate_cosine=NEAR_DUPLICATE_COSINE, sklearn=sklearn.__version__,
                  input_sha256=input_hashes, results=metrics)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + chr(10), encoding='utf-8')
    print('Completed:', output, flush=True)


if __name__ == '__main__':
    run(Path(os.environ.get('HRB_PILOT_OUTPUT', 'data/experiments/bulk_transfer_pilot_v1')))
