"""Map free-text classifier completions to one of the five role labels.

Version 2 of the parser. Version 1 (pilot_tables.clean + edit distance on the
whole first line) rejected correct answers followed by an explanation on the
same line, e.g. "Fear Monger. The post predominantly uses ...". Version 2
matches the label at the start of the answer within ``max_distance`` edits,
and otherwise accepts an answer that names exactly one label anywhere.
Unmatched or ambiguous answers stay invalid and remain in the denominator.
"""
import re
import string

ROLES = ('Anti-Vaccine Advocate', 'Fear Monger', 'Misinformation Spreader', 'Otherwise', 'Religious Conspiracy Theorist')
INVALID = '__invalid__'
# Zero-shot classification prompt shared by the open models (zero_shot.py) and GPT-4o (zero_shot_api.py).
INSTRUCTIONS = '''Classify the rhetorical role of the quoted post. Return only one of the exact category names below.
Misinformation Spreader: repeats or forwards an unverified rumor without predominantly using one of the other forms of rhetoric.
Religious Conspiracy Theorist: uses religious or supernatural beliefs to frame a conspiracy or a health claim.
Fear Monger: predominantly uses disproportionate alarm or exaggerated danger to provoke fear.
Anti-Vaccine Advocate: predominantly argues against vaccination or promotes distrust of vaccines or vaccination programs.
Otherwise: lacks the four listed misinformation-role framings.
Quoted post: '''
PARSER_VERSION = 'labels-v2'
_PREFIX = re.compile(r'^\s*(?:the\s+)?(?:category|role|answer|label|classification)\s*(?:is)?\s*[:\-]?\s*', re.I)
# Spelling variants seen in completions; canonical forms are matched as whole phrases.
_VARIANTS = {
    'Anti-Vaccine Advocate': (r'anti[\s-]*vaccine\s+advocate', r'anti[\s-]*vax(?:xer)?\s+advocate'),
    'Fear Monger': (r'fear[\s-]*monger(?:ing)?',),
    'Misinformation Spreader': (r'misinformation\s+spreader',),
    'Otherwise': (r'otherwise',),
    'Religious Conspiracy Theorist': (r'religious\s+conspiracy\s+theorist',),
}
_MENTION = {role: re.compile(r'(?<![\w-])(?:' + '|'.join(p) + r')(?![\w-])', re.I) for role, p in _VARIANTS.items()}


def canonical(text):
    text = text.casefold().replace('’', "'")
    text = re.sub(r'[^\w\s-]', ' ', text)
    return ' '.join(text.replace('-', ' ').split())


def edit_distance(a, b):
    last = list(range(len(b) + 1))
    for i, char in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, last[j] + 1, last[j - 1] + (char != other)))
        last = current
    return last[-1]


def first_line(text):
    return next((line.strip() for line in text.splitlines() if line.strip()), '')


def leading_match(text, labels=ROLES, max_distance=2):
    """Label whose words best match the same number of leading answer words."""
    words = canonical(_PREFIX.sub('', first_line(text)).strip(string.whitespace + '*"\'`')).split()
    scored = sorted((edit_distance(' '.join(words[:len(canonical(label).split())]), canonical(label)), label)
                    for label in labels)
    if words and scored[0][0] <= max_distance and scored[0][0] < scored[1][0]:
        return scored[0][1]
    return None


def mentioned(text, labels=ROLES):
    """The single label named anywhere in the answer, else None."""
    hits = {label for label in labels if _MENTION[label].search(text)}
    return hits.pop() if len(hits) == 1 else None


def parse(text, labels=ROLES, max_distance=2):
    return leading_match(text, labels, max_distance) or mentioned(text, labels) or INVALID
