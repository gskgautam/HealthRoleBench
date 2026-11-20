"""Paraphrased and adversarial rewrites of the evaluation set (Table 18).

1. `generate`: Mistral-7B-Instruct-v0.2 rewrites each of the 1,500 evaluation posts under two
   instructions (greedy decoding, so the rewrites are reproducible):
     paraphrased  same meaning, stance, claims, tone and intent, in different words;
     adversarial  same intent and stance, but without the surface cues a classifier can latch
                  onto (stock phrases, role keywords, exaggerated emotion, hashtags, share calls).
2. `validate`: a rewrite is kept only if it passes the same checks as curation:
     - names the target disease, 30-150 words, differs from the original;
     - the curation NLI judge (configs/filter_v2) still assigns the intended role:
       intended-role entailment >= 0.5, or for Otherwise, neutral entailment >= 0.5 and every
       positive-role entailment < 0.75.
     - no refusal or rewriting commentary (curation's refusal patterns plus rewrite-specific ones).
   Invalid rewrites are kept in the file with a reason but are not scored.

Run from PROJECT_ROOT:
  python -m healthrolebench.analysis.perturb generate --condition paraphrased
  python -m healthrolebench.analysis.perturb validate --condition paraphrased
"""
import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

from healthrolebench.common.artifacts import private_text_artifact

EVAL = Path('data/experiments/eval_set_v1')
REWRITER = 'mistral'
MAX_NEW_TOKENS = 256
INSTRUCTIONS = {
    'paraphrased': (
        'Rewrite the following social-media post in different words. Keep exactly the same meaning, stance, '
        'claims, tone and intent, the same disease, and a similar length. Do not add, remove or soften any claim. '
        'Output only the rewritten post.'),
    'adversarial': (
        'Rewrite the following social-media post so that the author\'s underlying intent and stance stay exactly the '
        'same, but the post is harder to classify from surface cues. Avoid stock phrases and keywords typical of this '
        'kind of post, remove exaggerated emotional wording, hashtags and calls to share, and express the same '
        'position in a calmer, more indirect way. Keep the same disease and a similar length. '
        'Output only the rewritten post.'),
}
ALIASES = {   # same disease anchors as curation (scripts/filter_v2/rules.py)
    'COVID-19': r'\b(?:covid(?:[ -]?19)?|coronavirus|sars[ -]?cov[ -]?2)\b',
    'HPV': r'\b(?:hpv|human papilloma(?:virus)?|cervical cancer|gardasil)\b',
    'Influenza': r'\b(?:influenza|flu|h1n1|h3n2)\b',
}
# Refusal and meta-text patterns, as in curation (scripts/filter_v2/rules.py), plus rewrite-specific commentary.
REFUSAL = re.compile(r"\b(?:i (?:cannot|can't|will not|won't|am unable to)(?! help (?:feel|think|but|wonder|notic))|i(?:'m| am) (?:not able|not comfortable|unable)|i must (?:decline|refuse))\b.{0,100}\b(?:create|generate|provide|fulfil|fulfill|assist|help|promote|spread|comply|request|rewrite|write)\b", re.I | re.S)
META = re.compile(r'\b(?:rewritten (?:post|version)|this rewrite|here is the rewrite|as an ai(?: language model)?|category\s*:|explanation\s*:)', re.I)
WRAPPER = re.compile(r'(?is)^\s*(?:sure|certainly|of course|here(?:\'s| is)[^\n:]*|rewritten post|rewrite)\b[^\n]*:\s*\n+')


def normalise(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def clean(raw):
    text = WRAPPER.sub('', raw.strip())
    text = re.split(r'\n\s*(?:note|explanation)\s*:', text, maxsplit=1, flags=re.I)[0]
    return text.strip().strip('"“”').strip()


def read(path):
    with path.open(newline='', encoding='utf-8') as source:
        return list(csv.DictReader(source))


def generate(condition):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from healthrolebench.models.zero_shot import CACHE, MODELS

    target = EVAL / f'rewrites_{condition}.csv'
    if target.exists():
        raise FileExistsError(target)
    posts = read(EVAL / 'eval_posts_private.csv')
    folder, revision = MODELS[REWRITER]
    snapshot = CACHE / ('models--' + folder) / 'snapshots' / revision
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, padding_side='left')
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, dtype=torch.float16).to('cuda:0').eval()
    with private_text_artifact(target) as handle:
        writer = csv.writer(handle)
        writer.writerow(('item_id', 'raw_completion'))
        for start in range(0, len(posts), 12):
            batch = posts[start:start + 12]
            prompts = [tok.apply_chat_template([{'role': 'user', 'content': INSTRUCTIONS[condition] + '\n\nPost: '
                                                 + json.dumps(row['post'], ensure_ascii=False)}],
                                               tokenize=False, add_generation_prompt=True) for row in batch]
            inputs = tok(prompts, padding=True, return_tensors='pt', add_special_tokens=False).to('cuda:0')
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            writer.writerows((row['item_id'], text) for row, text in zip(batch, texts))
            handle.flush()
            if start % 240 == 0:
                print('rewrite', condition, min(start + 12, len(posts)), '/', len(posts), flush=True)
    (EVAL / f'rewrites_{condition}_settings.json').write_text(json.dumps(dict(
        rewriter=folder, revision=revision, decoding='greedy', max_new_tokens=MAX_NEW_TOKENS, instruction=INSTRUCTIONS[condition],
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0)), indent=2) + chr(10))
    print('Completed rewrites', condition, len(posts), flush=True)


class Judge:
    """The frozen curation NLI judge (same model, revision, hypotheses, thresholds and truncation)."""

    def __init__(self, protocol_path=Path('configs/filter_v2/protocol.json'),
                 revision_path=Path('configs/filter_v2/judge_revision.json')):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from healthrolebench.models.zero_shot import CACHE

        self.torch = torch
        self.protocol = json.loads(protocol_path.read_text())['F2_role']
        pinned = json.loads(revision_path.read_text())
        snapshot = CACHE / ('models--' + pinned['model'].replace('/', '--')) / 'snapshots' / pinned['revision']
        self.tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(snapshot, local_files_only=True,
                                                                        dtype=torch.float16).to('cuda:0').eval()
        labels = {str(k).lower(): int(v) for k, v in self.model.config.label2id.items()}
        self.entailment = labels['entailment']
        self.manifest = dict(model=pinned['model'], revision=pinned['revision'], max_length=256, dtype='float16')

    def scores(self, pairs):
        results = []
        for start in range(0, len(pairs), 48):
            batch = pairs[start:start + 48]
            enc = self.tok([a for a, _ in batch], [b for _, b in batch], padding=True, truncation='only_first',
                           max_length=256, return_tensors='pt').to('cuda:0')
            with self.torch.inference_mode():
                results.extend(self.model(**enc).logits.float().softmax(-1)[:, self.entailment].cpu().tolist())
        return results

    def keeps_role(self, text, role):
        hypotheses = self.protocol['hypotheses']
        wanted = list(hypotheses) if role == 'Otherwise' else [role]
        values = dict(zip(wanted, self.scores([(text, hypotheses[r]) for r in wanted])))
        if role == 'Otherwise':
            ok = (values['Otherwise'] >= self.protocol['neutral_min_entailment'] and
                  max(v for k, v in values.items() if k != 'Otherwise') < self.protocol['neutral_max_positive_entailment'])
        else:
            ok = values[role] >= self.protocol['positive_min_entailment']
        return ok, values


def validate(condition):
    target = EVAL / f'perturbed_{condition}.csv'
    if target.exists():
        raise FileExistsError(target)
    key = {r['item_id']: r for r in read(EVAL / 'eval_key_private.csv')}
    originals = {r['item_id']: r['post'] for r in read(EVAL / 'eval_posts_private.csv')}
    rewrites = read(EVAL / f'rewrites_{condition}.csv')
    if {r['item_id'] for r in rewrites} != key.keys():
        raise ValueError('Rewrites do not cover the evaluation set')
    judge = Judge()
    rows, counts = [], {}
    for r in rewrites:
        item, text = r['item_id'], clean(r['raw_completion'])
        disease, role = key[item]['disease'], key[item]['intended_role']
        n = len(text.split())
        reason, scores = None, {}
        if REFUSAL.search(text) or META.search(text):
            reason = 'refusal_or_meta'
        elif not 30 <= n <= 150:
            reason = 'length'
        elif not re.search(ALIASES[disease], text, re.I):
            reason = 'missing_disease_anchor'
        elif normalise(text) == normalise(originals[item]):
            reason = 'unchanged'
        else:
            ok, scores = judge.keeps_role(text, role)
            reason = None if ok else 'role_not_preserved'
        counts[reason or 'valid'] = counts.get(reason or 'valid', 0) + 1
        rows.append((item, text, int(reason is None), reason or '', json.dumps(scores)))
    with private_text_artifact(target) as handle:
        writer = csv.writer(handle)
        writer.writerow(('item_id', 'rewrite', 'valid', 'reason', 'nli_scores'))
        writer.writerows(rows)
    by_role = {}
    for (item, _, valid, _, _) in rows:
        role = key[item]['intended_role']
        by_role.setdefault(role, [0, 0])
        by_role[role][0] += valid
        by_role[role][1] += 1
    (EVAL / f'perturbed_{condition}_summary.json').write_text(json.dumps(dict(
        condition=condition, outcomes=counts, valid_by_role={k: {'valid': v, 'n': n} for k, (v, n) in sorted(by_role.items())},
        judge=judge.manifest, rule='same as curation F2 (configs/filter_v2/protocol.json)'), indent=2) + chr(10))
    print('Validated', condition, counts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paraphrased and adversarial rewrites of the evaluation set (Table 18).')
    parser.add_argument('step', choices=('generate', 'validate'))
    parser.add_argument('--condition', required=True, choices=tuple(INSTRUCTIONS))
    args = parser.parse_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    (generate if args.step == 'generate' else validate)(args.condition)
