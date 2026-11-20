"""Draft-facing column names and explicit mappings to measured values."""
import json
import os
from collections import Counter, defaultdict
from statistics import mean

from healthrolebench.common.data import rows
from healthrolebench.common.labels import PARSER_VERSION
from healthrolebench.reporting.report_tables import verify_encoder_predictions

# Result locations (see README section 3). Pilot outputs (*_pilot_v1, zero_shot_audit_*) are kept for provenance.
ENC = 'encoder_tables_v1'      # Tables 7, 11, 13 (entropy), 14, 15: standard-encoder runs
EVALT = 'eval_tables_v1'
MIN_ROLE_SUPPORT = 30           # Table 18: roles with fewer label-preserving rewrites are shown as n=… instead of F1       # Tables 8, 9, 10, 13 (overlap), 18; Figure 8: 1,500-post evaluation set
CLASSIFIER_NAMES = {**{'gpt4o': 'GPT-4o'}, 'qwen': 'Qwen3.5-2B', 'llama2': 'LLaMA-2-7B', 'mistral': 'Mistral-7B-Instruct', 'gemma': 'Gemma-7B'}

MODELS = {
    'gpt4o': 'GPT-4o',
    'qwen': 'Qwen3.5-2B',
    'llama2': 'LLaMA-2-7B',
    'mistral': 'Mistral-7B-Instruct',
    'gemma': 'Gemma-7B',
}
MODEL_COLUMNS = tuple(MODELS.values())

HEADERS = {
    1: ('Role', 'Alignment', 'Potential Overlap'),
    2: ('Model', 'Initial Pairs', 'After Filter 1 (Pass / Fail)', 'After Filter 2 (Pass / Fail)', 'After Filter 3 (Pass / Fail)', 'Final Retained', 'Retention (%)'),
    3: ('Motivational Role', 'Samples', 'Label Agr. (%) ↑', 'IAA (%) ↑'),
    4: ('Role', 'Complete Lexical Markers ($L_r$)'),
    5: ('Role', 'Avg. Tokens', 'Avg. Sentences', 'Lexical Diversity (TTR)'),
    6: ('Class', 'Role', 'Disease', *MODEL_COLUMNS, 'Total Samples'),
    7: ('Disease', 'Role', 'Train (n)', 'Test (n)', 'TP', 'FP', 'FN', 'TN'),
    8: (r'Generator ($G_{\text{train}}$)', *MODEL_COLUMNS, 'Acc. ↑', 'Prec. ↑', 'Rec. ↑'),
    9: ('Generator ($G$)', *MODEL_COLUMNS, 'Mean F1 ↑'),
    10: ('Model', 'Zero-Shot F1 ↑', 'Fine-Tuned F1 ↑', 'Improvement (pp)'),
    11: (r'Train Disease ($d_{\text{train}}$)', 'COVID-19', 'HPV', 'Influenza', 'Acc. ↑', 'Prec. ↑', 'Rec. ↑'),
    12: ('Classifier', 'COVID-19', 'HPV', 'Influenza', 'Overall ↑'),
    13: ('Metric Type', 'MS', 'RCT', 'FM', 'AVA', 'OTH', 'Mean'),
    14: ('True Role', 'MS', 'RCT', 'FM', 'AVA'),
    15: ('Disease', 'Sample', 'True', 'Pred.'),
    16: ('True Role', 'MS', 'RCT', 'FM', 'AVA'),
    17: ('Setting', 'Macro-F1 ↑', 'Micro-F1 ↑', 'Ham. Loss ↓', 'Jaccard ↑'),
    18: ('Role', 'Baseline', 'Paraphrased', 'Cue-Masked', 'Adversarial'),
    19: ('Family', 'System A', 'System B', 'n', 'Δ Macro-F1 (pp)', '95% CI (pp)', 'p (Holm)'),
    20: ('Real training posts (k)', 'RoBERTa Macro-F1 ↑', 'HealthRoleBench-init Macro-F1 ↑', 'Δ (pp)', 'RoBERTa Acc. ↑', 'HealthRoleBench-init Acc. ↑'),
}
GROUPS = {
    8: (r'Classifier ($G_{\text{test}}$) — Macro-F1 (%) ↑', 'Aggregate Metrics (%)'),
    11: (r'Test Disease ($d_{\text{test}}$) — Macro-F1 (%) ↑', 'Aggregate Metrics (%)'),
}
ROLES = ('Misinformation Spreader','Religious Conspiracy Theorist','Fear Monger','Anti-Vaccine Advocate','Otherwise')
DISEASES = ('COVID-19','HPV','Influenza')
NA = 'N/A'
DIAG = '—'
NOTES = {
    19: 'Paired bootstrap (1,000 resamples of test posts, seed 42); two-sided p-values, Holm-corrected within each family. Three-seed systems use the per-post majority vote. 95% confidence intervals for each system are in the linked CSV.',
    20: 'Real data: CONSTRAINT@AAAI-2021 COVID-19 fake-news test split (2,140 posts). k = 0: the HealthRoleBench model applied directly (any misinformation role = fake). k > 0: binary fine-tuning on k real posts from plain RoBERTa or from the HealthRoleBench model; mean ± SD over 3 seeds.',
    2: 'Initial Pairs combines the scored bulk pool and logged top-up attempts across all roles. Filter 1 is pre-NLI screening. Filter 2 is the NLI role-context check. Filter 3 is selection within each role–disease quota, including accepted top-ups. Each failure count is relative to the preceding stage. Retention is Final Retained divided by Initial Pairs.',
    6: 'The columns show the five generators used in the reconstruction. Mistral uses the v0.2 checkpoint.',
    7: "Standard encoders (BERT, RoBERTa, DistilBERT) trained on two diseases and tested on the third; counts are means over 3 encoders × 3 seeds. Train (n) is the role's training count after leakage-group purging.",
    8: "Rows: generator of the test posts. Open models: frozen backbone + linear head trained on the other four generators, tested on all 12,000 posts of the row generator (mean of 3 seeds). GPT-4o: zero-shot on the row generator's 300 evaluation posts (no weights to train a head). Aggregates are means over classifiers.",
    9: 'Zero-shot, 1,500-post evaluation set (300 per generator). Diagonal entries (a model classifying its own posts) are excluded from means.',
    10: 'Same evaluation posts for both settings (off-diagonal generators). Fine-tuned = frozen backbone + trained linear head.',
    11: 'Standard encoders trained on one disease and tested on each other disease; means over 3 encoders × 3 seeds. Aggregates average the two test diseases.',
    12: 'Disease values use the verified available seeds. Overall appears once all three diseases have three seeds for that classifier.',
    13: 'Confusion entropy (nats) of the row-normalised confusion matrix, standard encoders, averaged over diseases and runs. Prediction overlap: per-role F1 agreement between zero-shot classifier pairs on posts both answered, averaged over pairs.',
    14: 'Percentage of all test posts of the true role predicted as each other role; standard encoders, pooled over diseases and runs.',
    15: 'The five most frequent confusions of the best encoder (RoBERTa, seed 42), one post each. True denotes the machine-curated intended role.',
    18: 'Per-role F1 averaged over the five zero-shot classifiers on the evaluation set. Paraphrased and adversarial use only rewrites that kept their role under the curation NLI judge; roles with fewer than 30 such rewrites show their count instead of an F1, and the macro average then covers the remaining roles.',
}


def load(root, name):
    path = root/name
    return list(rows(path)) if path.is_file() else []


def number(value, scale=100, digits=2):
    return NA if value in (None, '') else f'{float(value)*scale:.{digits}f}'


def means(pairs):
    buckets = defaultdict(list)
    for key, value in pairs:
        buckets[key].append(float(value))
    return {key:mean(bucket) for key,bucket in buckets.items()}


def curation_flow(root):
    project_data = root.parent
    scored_dir = project_data/'filter_runs/curation_v2'
    topup_dir = project_data/'topups_v2'
    report = {r['generator']:r for r in load(root,'corpus_tables_v1/table_02_curation.csv')}
    totals = [0, 0, 0, 0]
    result = []

    def measure(entries, decision_key, scores):
        screened = 0
        passed = Counter()
        for entry in entries:
            decision = entry[decision_key]
            evaluated = bool(scores(entry))
            if evaluated != (decision in ('pass','role_context_mismatch')):
                raise ValueError(f'Inconsistent NLI stage for decision {decision!r}')
            if evaluated:
                screened += 1
            if decision == 'pass':
                passed[entry['sample_id']] += 1
        return screened, passed

    def format_row(name, counts):
        initial, f1, f2, final = counts
        if not initial or not 0 <= final <= f2 <= f1 <= initial:
            raise ValueError(f'Invalid curation counts for {name}: {counts}')
        return (name,f'{initial:,}',f'{f1:,} / {initial-f1:,}',f'{f2:,} / {f1-f2:,}',
                f'{final:,} / {f2-final:,}',f'{final:,}',f'{final/initial:.2%}')

    for model,name in MODELS.items():
        directory = scored_dir/model
        bulk = list(rows(directory/'scored_candidates.csv'))
        with (topup_dir/f'{model}_attempts.jsonl').open(encoding='utf-8') as handle:
            topups = [json.loads(line) for line in handle if line.strip()]
        bulk_screened, bulk_eligible = measure(bulk,'curation_decision',lambda r:json.loads(r['nli_scores'] or '{}'))
        top_screened, top_eligible = measure(topups,'decision',lambda r:r['nli_scores'])
        bulk_kept = Counter(r['sample_id'] for r in rows(directory/'selected_machine_curated_draft.csv'))
        top_kept = Counter(r['sample_id'] for r in rows(topup_dir/f'{model}_candidates.csv'))
        if bulk_kept-bulk_eligible or top_kept != top_eligible or set(bulk_kept) & set(top_kept):
            raise ValueError(f'Final CSV IDs disagree with curation decisions for {model}')
        bulk_kept_n, top_kept_n = sum(bulk_kept.values()), sum(top_kept.values())
        final = bulk_kept_n+top_kept_n
        if (len(bulk),sum(bulk_eligible.values()),bulk_kept_n,top_kept_n,final) != tuple(int(report[model][field]) for field in
                ('bulk_scored_input','bulk_eligible','bulk_selected','accepted_topups','selected')):
            raise ValueError(f'Final counts disagree with the curation summary for {model}')
        counts = (len(bulk)+len(topups),bulk_screened+top_screened,sum(bulk_eligible.values())+sum(top_eligible.values()),final)
        totals = [a+b for a,b in zip(totals,counts)]
        result.append(format_row(name,counts))
    return result+[format_row('Aggregate Total',totals)]


def corpus_rows(table, root):
    if table == 1:
        return [(r['role'],r['alignment'],r['potential_overlap']) for r in load(root,'definition_tables_v1/table_01_taxonomy.csv')]
    if table == 2:
        return curation_flow(root)
    if table == 3:
        ratings = {r['role']:r for r in load(root,'human_audit_v1/table_03_human_validation.csv')}
        return [(role,ratings.get(role,{}).get('n',NA),number(ratings.get(role,{}).get('adjudicated_intended_agreement')),
                 number(ratings.get(role,{}).get('raw_agreement'))) for role in (*ROLES,'Overall')]
    if table == 4:
        markers = defaultdict(list)
        for row in load(root,'definition_tables_v1/table_04_lexical_markers.csv'):
            markers[row['role']].append(row['lexical_marker'])
        return [(role,', '.join(markers[role])) for role in ROLES[:4]]
    if table == 5:
        source = {r['role']:r for r in load(root,'corpus_tables_v1/table_05_linguistic.csv')}
        return [(role,number(source[role]['mean_whitespace_tokens'],1),number(source[role]['mean_sentence_segments'],1),
                 number(source[role]['mean_document_ttr'],1,3)) for role in ROLES if role in source]
    counts = {(r['generator'],r['disease'],r['intended_role']):int(r['selected'])
              for r in load(root,'corpus_tables_v1/table_06_distribution.csv')}
    result = []
    for role in ROLES:
        for disease in DISEASES:
            cells = [counts.get((model,disease,role)) for model in MODELS]
            result.append(('Otherwise' if role=='Otherwise' else 'Misinformation',role,disease,
                           *(value if value is not None else NA for value in cells),
                           sum(cells) if all(value is not None for value in cells) else NA))
    totals = [sum(counts[m,d,r] for d in DISEASES for r in ROLES)
              if all((m,d,r) in counts for d in DISEASES for r in ROLES) else None for m in MODELS]
    result.append(('Total (All Roles and Diseases)','','',*(v if v is not None else NA for v in totals),
                   sum(totals) if all(v is not None for v in totals) else NA))
    return result


def _cell(table, row_key, col_key, value):
    return DIAG if row_key == col_key else number(table.get((row_key, col_key)))


def transfer_rows(table, root):
    if table == 7:
        return [(r['disease'], r['role'], r['train_n'], r['test_n'], *(f"{float(r[k]):,.0f}" for k in ('tp', 'fp', 'fn', 'tn')))
                for r in load(root, f'{ENC}/table_07_confusion_counts.csv')]
    if table == 8:
        data = load(root, f'{EVALT}/table_08_cross_model.csv')
        f1 = {(r['held_generator'], r['classifier']): float(r['macro_f1']) for r in data}
        agg = defaultdict(list)
        for r in data:
            agg[r['held_generator']].append((float(r['accuracy']), float(r['macro_precision']), float(r['macro_recall'])))
        result = [(name, *(_cell(f1, g, c, None) for c in MODELS), *((number(mean(col)) for col in zip(*agg[g])) if agg[g] else (NA,) * 3))
                  for g, name in MODELS.items()]
        columns = [[v for (g, c), v in f1.items() if c == classifier] for classifier in MODELS]
        overall = [x for values in agg.values() for x in values]
        return result + [('Mean Across Models', *(number(mean(v)) if v else NA for v in columns),
                          *((number(mean(col)) for col in zip(*overall)) if overall else (NA,) * 3))]
    if table == 9:
        scores = {(r['generator'], r['classifier']): float(r['macro_f1']) for r in load(root, f'{EVALT}/table_09_zero_shot.csv')}
        off = {k: v for k, v in scores.items() if k[0] != k[1]}
        result = [(name, *(_cell(scores, g, c, None) for c in MODELS),
                   number(mean(v for (gg, _), v in off.items() if gg == g)) if any(gg == g for gg, _ in off) else NA)
                  for g, name in MODELS.items()]
        column = lambda c: number(mean(v for (_, cc), v in off.items() if cc == c)) if any(cc == c for _, cc in off) else NA
        return result + [('Mean across generators', *(column(c) for c in MODELS), number(mean(off.values())) if off else NA)]
    if table == 10:
        data = load(root, f'{EVALT}/table_10_zero_shot_vs_head.csv')
        result = [(MODELS[r['classifier']], number(r['zero_shot_macro_f1']), number(r['head_macro_f1']), number(r['delta_pp'], 1))
                  for r in data]
        if result:
            result.append(('Mean', *(number(mean(float(r[k]) for r in data), 1 if k == 'delta_pp' else 100)
                                     for k in ('zero_shot_macro_f1', 'head_macro_f1', 'delta_pp'))))
        return result
    if table == 11:
        data = load(root, f'{ENC}/table_11_cross_disease.csv')
        cell = {(r['train_disease'], r['test_disease']): r for r in data}
        result = []
        for d in DISEASES:
            rows_d = [cell[d, t] for t in DISEASES if (d, t) in cell]
            stats = [number(mean(float(r[k]) for r in rows_d)) if rows_d else NA for k in ('accuracy', 'macro_precision', 'macro_recall')]
            result.append((d, *(DIAG if d == t else number(cell[d, t]['macro_f1']) if (d, t) in cell else NA for t in DISEASES), *stats))
        test_mean = [number(mean(float(cell[d, t]['macro_f1']) for d in DISEASES if (d, t) in cell)) if any((d, t) in cell for d in DISEASES) else NA
                     for t in DISEASES]
        overall = [number(mean(float(r[k]) for r in data)) if data else NA for k in ('accuracy', 'macro_precision', 'macro_recall')]
        return result + [('Mean Across Diseases', *test_mean, *overall)]
    scores = defaultdict(list)
    for path in sorted((root / 'encoder_transfer_v1').glob('*/*/seed_*/metrics.json')):
        data = json.loads(path.read_text())
        scores[path.parents[2].name, data['disease']].append(verify_encoder_predictions(path.parent, data))
    result = []
    for model, name in (('bert', 'BERT-base-uncased'), ('roberta', 'RoBERTa-base'), ('distilbert', 'DistilBERT-base')):
        buckets = [scores[model, d] for d in DISEASES]
        result.append((name, *(number(mean(b)) if b else NA for b in buckets),
                       number(mean(mean(b) for b in buckets)) if all(len(b) == 3 for b in buckets) else NA))
    return result


def analysis_rows(table, root):
    if table == 13:
        entropy = {r['role']: float(r['confusion_entropy_nats']) for r in load(root, f'{ENC}/table_13_confusion_entropy.csv')}
        overlap = means((r['role'], r['agreement_f1']) for r in load(root, f'{EVALT}/table_13_prediction_overlap.csv'))
        return [(label, *(number(values.get(role), scale) for role in ROLES), number(mean(values.values()), scale) if values else NA)
                for label, values, scale in (('Confusion Entropy (CE) ↓', entropy, 1), ('Cross-Model F1 Overlap ↑', overlap, 100))]
    if table == 14:
        share = {(r['true_role'], r['predicted_role']): float(r['percent_of_true_role']) for r in load(root, f'{ENC}/table_14_error_distribution.csv')}
        return [(role, *(DIAG if role == p else number(share.get((role, p)), 1) for p in ROLES[:4])) for role in ROLES]
    if table == 15:
        short = {'Misinformation Spreader': 'MS', 'Religious Conspiracy Theorist': 'RCT', 'Fear Monger': 'FM',
                 'Anti-Vaccine Advocate': 'AVA', 'Otherwise': 'OTH'}
        return [(r['disease'], ' '.join(r['sample'].split()), short[r['true_role']], short[r['predicted_role']])
                for r in load(root, f'{ENC}/table_15_qualitative_errors.csv')]
    if table == 16:
        overlap = {(r['role_1'], r['role_2']): r['jaccard_percent'] for r in load(root, 'corpus_tables_v1/table_16_lexical_overlap.csv')}
        return [(role, *(DIAG if role == p else number(overlap.get((role, p)), 1) for p in ROLES[:4])) for role in ROLES]
    if table == 17:
        data = load(root, 'human_audit_v1/table_17_multilabel_evaluation.csv')
        return [(r['setting'], *(number(r[k]) for k in ('macro_f1', 'micro_f1', 'hamming_loss', 'samples_jaccard'))) for r in data] or [
            (name, NA, NA, NA, NA) for name in ('Single-label (original)', 'Multi-label (binary relevance)',
                                                'Multi-label (threshold-tuned)', 'Soft-label (annotator distribution)')]
    if table == 19:
        names = {'gpt4o': 'GPT-4o', 'qwen': 'Qwen3.5-2B', 'llama2': 'LLaMA-2-7B', 'mistral': 'Mistral-7B-Instruct',
                 'gemma': 'Gemma-7B', 'bert': 'BERT', 'roberta': 'RoBERTa', 'distilbert': 'DistilBERT'}
        pretty = lambda x: ' '.join(names.get(w, w) for w in x.split(' '))
        return [(r['family'], pretty(r['system_a']), pretty(r['system_b']), r['n'], number(r['difference']),
                 f"[{number(r['ci_low'])}, {number(r['ci_high'])}]", f"{float(r['p_holm']):.3g}")
                for r in load(root, 'significance_v1/table_19_paired_tests.csv')]
    if table == 20:
        out = []
        for r in load(root, 'domain_adapt_v1/table_20_domain_adaptation.csv'):
            if r['setting'] == 'zero-shot transfer':
                out.append(('0 (transfer)', NA, number(r['hrb_macro_f1']), NA, NA, number(r['hrb_accuracy'])))
            else:
                sd = lambda m, s: f"{number(m)} ± {number(s)}"
                out.append((r['k_real_train'], sd(r['roberta_macro_f1'], r['roberta_sd']), sd(r['hrb_macro_f1'], r['hrb_sd']),
                            number(float(r['hrb_macro_f1']) - float(r['roberta_macro_f1'])), number(r['roberta_accuracy']),
                            number(r['hrb_accuracy'])))
        return out
    data = load(root, f'{EVALT}/table_18_stress.csv')
    f1 = means(((r['role'], r['condition']), r['f1']) for r in data)
    support = {(r['role'], r['condition']): int(r['role_support']) for r in data}
    conditions = ('original', 'paraphrased', 'cue-masked', 'adversarial')
    reliable = lambda role, c: support.get((role, c), 0) >= MIN_ROLE_SUPPORT
    cell = lambda role, c: (number(f1[role, c]) if reliable(role, c) else f'n={support[role, c]}') if (role, c) in f1 else NA
    result = [(role, *(cell(role, c) for c in conditions)) for role in ROLES[:4]]
    macro = []
    for c in conditions:
        values = [f1[role, c] for role in ROLES[:4] if (role, c) in f1 and reliable(role, c)]
        macro.append(NA if not values else number(mean(values)) + ('' if len(values) == 4 else f' ({len(values)} roles)'))
    return result + [('Macro Avg.', *macro)]


def markdown_rows(table, root):
    columns = HEADERS[table]
    data = corpus_rows(table,root) if table<=6 else transfer_rows(table,root) if table<=12 else analysis_rows(table,root)
    if not data:
        data = [(NA,)*len(columns)]
    if any(len(row)!=len(columns) for row in data):
        raise ValueError(f'Table {table} does not match manuscript column count')
    out = []
    if table in GROUPS:
        out.append(f'**{GROUPS[table][0]}** · **{GROUPS[table][1]}**')
    if table in NOTES:
        out.append(NOTES[table])
    out.extend(('', '| '+' | '.join(columns)+' |', '| '+' | '.join('---' for _ in columns)+' |'))
    out.extend('| '+' | '.join(str(cell).replace('|',chr(92)+'|') for cell in row)+' |' for row in data)
    return out
