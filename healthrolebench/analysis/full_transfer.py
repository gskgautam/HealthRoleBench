import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

ROOT = Path('data')
OUT = ROOT / 'experiments' / 'full_transfer_pilot_v1'
CORPUS = ROOT / 'experiments' / 'corpus_tables_v1'
MODELS = ('gpt4o', 'qwen', 'llama2', 'mistral', 'gemma')
SIMILARITY = .80


def read(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    records = []
    for model in MODELS:
        bulk = ROOT/'filter_runs'/'curation_v2'/model/'selected_machine_curated_draft.csv'
        topup = ROOT/'topups_v2'/f'{model}_candidates.csv'
        for stage, path in (('bulk',bulk), ('topup',topup)):
            records.extend((model, r['sample_id'], r['disease'], r['intended_role'], stage, r['text']) for r in read(path))
    assert len(records) == 60000
    indices = {(r[0],r[1]): i for i,r in enumerate(records)}
    assert len(indices) == len(records)
    generators, sample_ids, diseases, labels, stages, text = np.asarray(records, dtype=object).T
    classes = sorted(set(labels))
    parent = list(range(len(records)))
    sizes = [1] * len(parent)

    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def merge(i, j):
        i, j = find(i), find(j)
        if i == j:
            return
        if sizes[i] < sizes[j]:
            i, j = j, i
        parent[j] = i
        sizes[i] += sizes[j]

    links = read(CORPUS/'topup_exemplar_links.csv')
    assert len(links) == 33282
    for link in links:
        model = link['generator']
        merge(indices[model,link['topup_id']], indices[model,link['exemplar_id']])
    trigrams = normalize(CountVectorizer(ngram_range=(3,3), binary=True, dtype=np.float32).fit_transform(text))
    transposed = trigrams.T.tocsr()
    lexical_pairs = 0
    for start in range(0,len(records),128):
        similarities = trigrams[start:start+128] @ transposed
        first, second = (similarities >= SIMILARITY).nonzero()
        for a,b in zip(first+start, second):
            if a < b:
                lexical_pairs += 1
                merge(int(a), int(b))
    groups = np.asarray([find(i) for i in range(len(records))])
    del trigrams, transposed
    OUT.mkdir(parents=True)
    print('Near-duplicate pairs:', lexical_pairs, 'groups:', len(set(groups)), 'largest group:', max(Counter(groups).values()), flush=True)
    with (OUT/'groups.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('generator','sample_id','disease','role','stage','group'))
        writer.writerows((*record[:-1],int(group)) for record,group in zip(records,groups))
    folds = []
    for protocol, domains in (('generator',generators), ('disease',diseases)):
        for test_domain in sorted(set(domains)):
            folds.append((protocol, 'all_other', test_domain, domains != test_domain, domains == test_domain))
    for source in sorted(set(diseases)):
        for target in sorted(set(diseases) - {source}):
            folds.append(('disease_pair', source, target, diseases == source, diseases == target))
    results = []
    with (OUT/'predictions.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('protocol','source','target','generator','sample_id','disease','stage','role','prediction'))
        for protocol, source, target, train_mask, test_mask in folds:
            test = np.flatnonzero(test_mask)
            train_all = np.flatnonzero(train_mask)
            train = train_all[~np.isin(groups[train_all],groups[test])]
            assert not (set(groups[test]) & set(groups[train]))
            assert set(labels[train]) == set(labels[test]) == set(classes)
            estimator = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True),
                                      LinearSVC(class_weight='balanced', random_state=42))
            estimator.fit(text[train],labels[train])
            prediction = estimator.predict(text[test])
            report = classification_report(labels[test],prediction,labels=classes,output_dict=True,zero_division=0)
            matrix = confusion_matrix(labels[test],prediction,labels=classes)
            assert matrix.sum() == len(test)
            excluded = len(train_all)-len(train)
            results.append(dict(protocol=protocol, source=source, target=target, train_n=len(train), test_n=len(test),
                                excluded_train=excluded, accuracy=accuracy_score(labels[test],prediction), report=report,
                                confusion_matrix=matrix.tolist()))
            writer.writerows((protocol,source,target,*records[j][0:3],stages[j],labels[j],p) for j,p in zip(test,prediction))
            print(protocol,source,target,'test',len(test),'purged',excluded,'macro-F1', round(report['macro avg']['f1-score'],4), flush=True)
    manifest = json.loads((CORPUS/'manifest.json').read_text())
    data = dict(status='pilot against machine-curated intended labels', input_sha256=manifest['input_sha256'],
                lexical_cosine=SIMILARITY, lexical_pairs=lexical_pairs, exemplar_links=len(links), classes=classes, results=results)
    (OUT/'metrics.json').write_text(json.dumps(data,indent=2)+chr(10),encoding='utf-8')
    print('Completed:',OUT,flush=True)

if __name__ == '__main__':
    main()
