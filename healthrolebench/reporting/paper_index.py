import argparse
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

from healthrolebench.reporting.draft_tables import ENC, EVALT, markdown_rows
from healthrolebench.reporting.report_tables import encoder_summary

# (number, title, description, result files under data/experiments/, pending reason). Only these
# aggregate files are copied into the Git tree. A table is Complete when all its files exist.
HUMAN = 'Needs the two independent annotations and adjudication of the 450-post audit (human_audit.py)'
TABLES = (
    (1,'Motivational-role taxonomy','Approved taxonomy; counts from the corpus.',('definition_tables_v1/table_01_taxonomy.csv',),None),
    (2,'Curation and corpus flow','Candidate, screening, NLI and quota counts, including top-ups.',('corpus_tables_v1/table_02_curation.csv',),None),
    (3,'Human validation','Agreement of adjudicated human labels with intended roles; inter-annotator agreement.',('human_audit_v1/table_03_human_validation.csv',),HUMAN),
    (4,'Role-specific lexical markers','Exact curation lexicon (a diagnostic warning in v2).',('definition_tables_v1/table_04_lexical_markers.csv',),None),
    (5,'Linguistic characteristics','Token, sentence and type–token statistics by role.',('corpus_tables_v1/table_05_linguistic.csv',),None),
    (6,'Dataset distribution','All 75 generator × disease × role cells.',('corpus_tables_v1/table_06_distribution.csv',),None),
    (7,'Confusion counts','Standard encoders, held-out disease.',(f'{ENC}/table_07_confusion_counts.csv',),None),
    (8,'Cross-model classification','Frozen-backbone heads on full held-out generators; GPT-4o zero-shot.',(f'{EVALT}/table_08_cross_model.csv',),None),
    (9,'Zero-shot cross-model classification','Five classifiers on the 1,500-post evaluation set.',(f'{EVALT}/table_09_zero_shot.csv',),None),
    (10,'Zero-shot versus fine-tuned','Paired on the evaluation set.',(f'{EVALT}/table_10_zero_shot_vs_head.csv',),None),
    (11,'Cross-disease transfer','Standard encoders trained on one disease.',(f'{ENC}/table_11_cross_disease.csv',),None),
    (12,'Standard encoder baselines','BERT, RoBERTa, DistilBERT; 3 held-out diseases × 3 seeds.',('encoder_transfer_v1/table_12_encoder_macro_f1.csv',),None),
    (13,'Confusion entropy and prediction overlap','Encoder confusion entropy; zero-shot agreement.',(f'{ENC}/table_13_confusion_entropy.csv',f'{EVALT}/table_13_prediction_overlap.csv'),None),
    (14,'Per-role error distribution','Standard encoders, held-out disease.',(f'{ENC}/table_14_error_distribution.csv',),None),
    (15,'Qualitative errors','Most frequent encoder confusions, one post each.',(f'{ENC}/table_15_qualitative_errors.csv',),None),
    (16,'Per-role lexical overlap','Jaccard overlap of the top smoothed log-odds n-gram sets.',('corpus_tables_v1/table_16_lexical_overlap.csv',),None),
    (17,'Multi-label evaluation','Four label formulations on the human-audited posts (multilabel.py).',('human_audit_v1/table_17_multilabel_evaluation.csv',),HUMAN),
    (18,'Prompt sensitivity and stress tests','Original, paraphrased, cue-masked and adversarial posts.',(f'{EVALT}/table_18_stress.csv',),None),
    (19,'Confidence intervals and significance tests','Bootstrap 95% CIs and Holm-corrected paired tests for Tables 9, 10, 12, 18 and 20.',('significance_v1/table_19_paired_tests.csv','significance_v1/table_19_confidence_intervals.csv'),None),
    (20,'Domain adaptation to real posts','HealthRoleBench as a starting point for detecting misinformation in real COVID-19 posts (CONSTRAINT@AAAI-2021).',('domain_adapt_v1/table_20_domain_adaptation.csv',),None),
)
FIGURES = (
    (5,'Intended-role threshold sensitivity','threshold_f1_v1/figure_5',('threshold_f1_v1/figures_5_7_f1.csv','threshold_f1_v1/figure_5_retention.csv')),
    (6,'Otherwise screening sensitivity','threshold_f1_v1/figure_6',('threshold_f1_v1/figures_5_7_f1.csv','threshold_f1_v1/figure_6_retention.csv')),
    (7,'Lexical threshold sensitivity','threshold_f1_v1/figure_7',('threshold_f1_v1/figures_5_7_f1.csv','threshold_f1_v1/figure_7_retention.csv')),
    (8,'Label-parser tolerance (λ)',f'{EVALT}/figure_8',(f'{EVALT}/figure_8.csv',)),
)


def link_asset(source, assets):
    if not source.is_file():
        return None
    target = assets/source.name
    with NamedTemporaryFile(dir=assets, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        if source.suffix == ".csv":
            temporary.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            shutil.copyfile(source,temporary)
        temporary.chmod(0o644)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return f'[{target.name}](paper_assets/{target.name})'


def figure_preview(pdf, assets):
    executable = shutil.which('pdftoppm')
    if executable is None:
        return None
    target = assets / (pdf.stem + '.png')
    with NamedTemporaryFile(dir=assets, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            subprocess.run([executable, '-singlefile', '-png', '-r', '125', str(pdf)], stdout=handle, check=True)
            temporary.chmod(0o644)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return f'![{pdf.stem}](paper_assets/{target.name})'


def render(root, repo):
    experiments = root/'data/experiments'
    encoder_dir = experiments/'encoder_transfer_v1'
    completed_runs = sorted(encoder_dir.glob('*/*/seed_*/metrics.json'))
    if len(completed_runs) == 27 and not (encoder_dir/'table_12_encoder_macro_f1.csv').is_file():
        encoder_summary(root)
    assets = repo/'paper_assets'
    assets.mkdir(exist_ok=True)
    out = ['# HealthRoleBench tables and figures','','Tables follow manuscript numbering. Results are scored against machine-curated intended labels unless independent human ratings are explicitly identified. Non-model column headings and order follow the draft. Model rows and columns identify the models used in the reconstruction. N/A marks unavailable values or an unmatched protocol. Linked CSVs retain the detailed result schemas.','','## Tables']
    for number,title,note,files,pending in TABLES:
        links = [link_asset(experiments/file,assets) for file in files]
        complete = all(links)
        status = 'Complete' if complete else 'Pending: ' + (pending or 'run the steps in README section 3')
        out.extend(('',f'### Table {number} — {title}','',f'**{status}.** {note}'))
        if any(links):
            out.extend(('', ' · '.join(link for link in links if link)))
        out.extend(('', *markdown_rows(number, experiments)))
    out.extend(('', '## Figures'))
    for number,title,stem,data in FIGURES:
        pdf = experiments/(stem+'.pdf')
        links = [link_asset(path,assets) for path in (pdf, *(experiments/d for d in data))]
        status = 'Complete' if all(links) else 'Pending: run the steps in README section 3'
        out.extend(('',f'### Figure {number} — {title}','',f'**{status}.**'))
        if any(links):
            out.extend(('', ' · '.join(link for link in dict.fromkeys(links) if link)))
        image = figure_preview(pdf, assets) if pdf.is_file() else None
        if image:
            out.extend(('', image))
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=repo, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(chr(10).join(out)+chr(10))
    temporary.chmod(0o644)
    temporary.replace(repo/'EXPERIMENTS.md')
    print('Wrote ordered Tables 1–20 and Figures 5–8:',repo/'EXPERIMENTS.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refresh the ordered manuscript results index and aggregate assets.')
    parser.add_argument('--project-root',required=True,type=Path)
    parser.add_argument('--repo-root',required=True,type=Path)
    args = parser.parse_args()
    render(args.project_root.resolve(),args.repo_root.resolve())
