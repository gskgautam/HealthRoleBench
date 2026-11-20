import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from sklearn.metrics import accuracy_score, f1_score

from healthrolebench.common.data import rows
from healthrolebench.curation.definition_tables import write_csv
from healthrolebench.common.labels import PARSER_VERSION
from healthrolebench.reporting.stress_tables import MODELS, read_predictions


def export(root, run='zero_shot_audit_v2'):
    experiments = root/'data/experiments'
    probe = experiments/'frozen_head_audit_v1'
    zero = experiments/run
    out = probe/PARSER_VERSION
    out.mkdir(exist_ok=True)
    truth = {r['audit_id']:r for r in rows(experiments/'corpus_tables_v1/audit_key_private.csv')}
    labels = sorted({r['intended_role'] for r in truth.values()})
    generators = sorted({r['generator'] for r in truth.values()})
    seeds = ('42','43','44')
    cross, comparison = [], []
    for model in MODELS:
        prompted = read_predictions(zero/f'{model}.csv',truth.keys(),labels)
        predictions = defaultdict(dict)
        for r in rows(probe/f'{model}.csv'):
            seed, item = r['seed'],r['audit_id']
            if seed not in seeds or item not in truth or item in predictions[seed] or truth[item]['generator'] != r['held_generator']:
                raise ValueError(f'Invalid prediction provenance for {model}: {item}')
            if r['prediction'] not in labels:
                raise ValueError(f'Invalid classifier label for {model}: {item}')
            predictions[seed][item] = r['prediction']
        if any(predictions[s].keys() != truth.keys() for s in seeds):
            raise ValueError(f'Incomplete predictions for {model}')
        baseline, per_seed = [], defaultdict(list)
        for generator in generators:
            if generator == model:
                continue
            ids = [item for item,r in truth.items() if r['generator'] == generator]
            gold = [truth[i]['intended_role'] for i in ids]
            baseline.append(f1_score(gold,[prompted[i] for i in ids],labels=labels,average='macro',zero_division=0))
            f1, accuracy = [], []
            for seed in seeds:
                pred = [predictions[seed][i] for i in ids]
                f1.append(f1_score(gold,pred,labels=labels,average='macro',zero_division=0))
                accuracy.append(accuracy_score(gold,pred))
                per_seed[seed].append(f1[-1])
            cross.append((generator,model,len(ids),mean(f1),stdev(f1),mean(accuracy)))
        averages = [mean(per_seed[seed]) for seed in seeds]
        comparison.append((model,len(baseline),mean(baseline),mean(averages),stdev(averages),100*(mean(averages)-mean(baseline))))
    write_csv(out/'table_08_cross_generator_frozen_head_pilot.csv',('held_generator','classifier','n','macro_f1','macro_f1_sd','accuracy'),cross)
    write_csv(out/'table_10_zero_shot_vs_frozen_head_pilot.csv',('classifier','paired_generator_folds','zero_shot_macro_f1','frozen_head_macro_f1','frozen_head_sd','delta_pp'),comparison)
    print('Saved paired Tables 8 and 10 pilot results.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Export cross-generator probe and matched zero-shot comparisons.')
    parser.add_argument('--project-root',required=True,type=Path)
    parser.add_argument('--run',default='zero_shot_audit_v2',help='Zero-shot run used as the paired baseline')
    args = parser.parse_args()
    export(args.project_root.resolve(), args.run)
