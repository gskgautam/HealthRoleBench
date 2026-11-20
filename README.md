# HealthRoleBench

Code and results for *Who Spreads Misinformation? Benchmarking LLMs on Role-Aware Health Narratives* (under review).

HealthRoleBench is a controlled benchmark of 60,000 synthetic posts on COVID-19, HPV, and influenza vaccination. Five generators (GPT-4o, Qwen3.5-2B, LLaMA-2-7B-chat, Mistral-7B-Instruct-v0.2, Gemma-7B) each contribute 12,000 posts. Every post is labeled with one of four motivational roles (Misinformation Spreader, Religious Conspiracy Theorist, Fear Monger, Anti-Vaccine Advocate) or the Otherwise class.

## Repository map

| Path | Contents |
|---|---|
| `healthrolebench/common/` | Corpus loader, leakage-aware splits, label parser, evaluation items |
| `healthrolebench/curation/` | Corpus statistics, leakage groups, threshold sweeps, evaluation set, audit sample |
| `healthrolebench/models/` | Fine-tuned encoders, linear heads on frozen LLMs, zero-shot classifiers (open and GPT-4o) |
| `healthrolebench/analysis/` | Stress tests, bootstrap significance, multi-label evaluation, transfer to real posts |
| `healthrolebench/reporting/` | Table exports and the pgfplots figures used in the paper |
| `scripts/` | GPU lane launchers for encoders, linear heads, and zero-shot runs |
| `configs/filter_v1/` | Filter protocol (NLI thresholds, role lexicons) and pinned encoder revisions |
| `results/tables/` | Aggregate CSV for every table and figure in the paper |
| `results/figures/` | Generated LaTeX for the paper figures |
| `tests/` | CPU unit tests (no private data) |
| `reproduce.sh` | Runs every stage in dependency order |

Generation prompts and generation scripts are not part of this repository (see [Responsible Use](#responsible-use)). API keys, model weights, raw generations, and annotation forms are also excluded.

## 1. Prepare the Environment

Verified with Python 3.10.12 on Linux with NVIDIA RTX 3090 GPUs and CUDA 12.8. Set the repository location once, then run all commands from the repository root.

```bash
export ANONYMOUS_REPOSITORY_URL=https://github.com/OWNER/REPOSITORY.git
git clone "$ANONYMOUS_REPOSITORY_URL" HealthRoleBench
cd HealthRoleBench
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt && pip install -e . --no-deps
python -m unittest discover -s tests
```

CPU-only steps (table exports, significance tests) need only `pip install -e .`.

## 2. Install the Corpus

The corpus is included in this repository as `data/healthrolebench_v1.zip` (posts, labels, and leakage groups only). Verify and extract it.

```bash
echo "70a2b0a61fa84dd50d92aea64ff4e92ffe1fa40d3d2f90dda7f7057d406f3a7d  data/healthrolebench_v1.zip" | sha256sum -c
unzip data/healthrolebench_v1.zip -d data
python3 -c "from pathlib import Path; from healthrolebench.common.data import load_corpus; r,_=load_corpus(Path('.')); print(len(r))"
```

`data/healthrolebench_v1.csv` has one row per post with the columns `generator`, `sample_id`, `disease`, `label`, `stage` (`bulk` or `topup`), `group` (leakage group used by every split), and `text`. Expected SHA-256 of the CSV is `45037b80a64194a608aec2fb08a883a45e5644b42686be64f3ef153395f0bcaf`. Expected counts are 60,000 posts, with 7,500 per role and 30,000 Otherwise.

The loader uses this file whenever it is present. Curation-stage outputs (filter scores, threshold sweeps, audit sample, evaluation set) are derived from the full curation snapshot, which contains the generation prompts and is shared with researchers on request under the same license.

## 3. Reproduce the Paper

`reproduce.sh PROJECT_ROOT [STAGE ...]` runs the stages below in order. Every step writes its outputs once to `data/experiments/` and refuses to overwrite them, so the script can be resumed after an interruption.

| Paper item | Stage | Command | Hardware |
|---|---|---|---|
| Tables II and IV, Fig. 3(b) | `corpus` | `python -m healthrolebench.curation.corpus_tables` | CPU |
| Tables I and III | `corpus` | `python -m healthrolebench.curation.definition_tables --project-root .` | CPU |
| Leakage groups | `groups` | `python -m healthrolebench.curation.exemplar_links && python -m healthrolebench.analysis.full_transfer` | CPU |
| Fig. 8(b, c) | `thresholds` | `python -m healthrolebench.curation.threshold_f1 --project-root .` | CPU |
| Evaluation set (1,500 posts) | `evalset` | `python -m healthrolebench.curation.eval_set --project-root .` | CPU |
| Fig. 4 | `encoders` | `scripts/run_encoder_lane.sh . 0 bert roberta distilbert` (repeat with `PROTOCOL=source`) | GPU |
| Fig. 8(a) rewrites | `rewrites` | `python -m healthrolebench.analysis.perturb generate --condition paraphrased`, then `validate` (same for `adversarial`) | GPU |
| Fig. 6(b), open models | `zeroshot` | `ITEMS=eval scripts/run_zero_shot_lane.sh . 0 qwen llama2 mistral gemma` | GPU |
| Fig. 6(b), GPT-4o | `gpt4o` | `python -m healthrolebench.models.zero_shot_api --items eval --condition original --max-usd 1` | API, about US$0.82 per condition |
| Fig. 6(a, c) | `probes` | `scripts/run_probe_full_lane.sh . 1` | GPU |
| Fig. 3(d) | `domain` | `python -m healthrolebench.analysis.domain_adapt all --project-root .` | GPU |
| Fig. 7 and Fig. 8(a, d) | `tables` | `python -m healthrolebench.reporting.encoder_tables --project-root . && python -m healthrolebench.reporting.eval_tables --project-root .` | CPU |
| Fig. 6(d) | `tables` | `python -m healthrolebench.analysis.significance --project-root .` | CPU |
| Fig. 3(a, c) | after annotation | `python -m healthrolebench.curation.human_audit score --project-root .` and `python -m healthrolebench.analysis.multilabel --project-root .` | CPU |
| Paper figures | `tables` | `python -m healthrolebench.reporting.paper_figures --assets results/tables --out results/figures` | CPU |

Fig. 3(d) also needs the CONSTRAINT@AAAI-2021 COVID-19 corpus (Patwa et al., 2021) in `data/external/constraint_covid19/{train,val,test}.csv`; the experiments used the Hugging Face copy `nanyy1025/covid_fake_news` at revision `079d310163b36c9a72fe327d905cf21a21b8d5a6`. The GPT-4o stage reads `OPENAI_API_KEY` from the environment; never store keys in the repository.

## 4. Settings

- **Splits.** Posts linked through top-up examples or with trigram cosine similarity of at least 0.8 share a `group` and never cross a split.
- **Fine-tuned encoders.** BERT, DistilBERT, RoBERTa; learning rate 1e-5, batch size 32, three epochs, seeds 42–44.
- **Linear heads.** Frozen final-layer representations; full-batch AdamW, learning rate 0.01, class-weighted loss, steps selected on validation.
- **Zero-shot classifiers.** Deterministic decoding, at most 48 new tokens, parser `labels-v2` with edit-distance tolerance 2. A classifier is not scored on posts from its own generator.
- **Statistics.** 95% percentile bootstrap intervals (1,000 resamples), paired bootstrap tests, Holm correction within each family.
- **Model revisions.** Pinned in `healthrolebench/models/zero_shot.py`, `healthrolebench/models/train_encoder.py`, and `configs/filter_v1/encoder_revisions.json`.

## Responsible Use

HealthRoleBench contains synthetic health misinformation. It is released for research on detecting and mitigating misinformation under the terms in [LICENSE](LICENSE), which prohibit using the corpus to generate misinformation, to train models that generate it, or to disseminate it. Any reproduced post must be labeled as synthetic misinformation. Generation prompts and generation scripts are withheld for this reason.

## Citation

```bibtex
@article{anonymous2026healthrolebench,
  title   = {Who Spreads Misinformation? Benchmarking LLMs on Role-Aware Health Narratives},
  author  = {Anonymous Authors},
  note    = {Under review},
  year    = {2026}
}
```
