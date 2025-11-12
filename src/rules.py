import re
from typing import List
from rapidfuzz import process, fuzz

# ---------------------------------------------------------
# EMAIL NORMALIZATION
# ---------------------------------------------------------
EMAIL_TOKEN_PATTERNS = [
    (r'\b\(?(at|@)\)?\b', '@'),
    (r'\b(dot|dat)\b', '.'),
    (r'\s*@\s*', '@'),
    (r'\s*\.\s*', '.')
]

def collapse_spelled_letters(s: str) -> str:
    # Collapse spaced letters like "g m a i l" → "gmail"
    return re.sub(r'(?:\b([a-zA-Z])\s*){3,}', 
                  lambda m: ''.join(re.findall(r'[a-zA-Z]', m.group(0))), s)

def normalize_email_tokens(s: str) -> str:
    s2 = collapse_spelled_letters(s.lower())
    for pat, rep in EMAIL_TOKEN_PATTERNS:
        s2 = re.sub(pat, rep, s2, flags=re.IGNORECASE)
    # Clean " gmailcom" or "gmail com"
    s2 = re.sub(r'(@[a-z0-9]+)\s*dot\s*(com|co|in|org)\b', r'\1.\2', s2)
    s2 = re.sub(r'(@[a-z0-9]+)\s*(com|in|org|co\.in)\b', r'\1.\2', s2)
    s2 = re.sub(r'(\w)gmailcom\b', r'\1@gmail.com', s2)
    s2 = re.sub(r'(\w)yahoocom\b', r'\1@yahoo.com', s2)
    s2 = re.sub(r'(\w)outlookcom\b', r'\1@outlook.com', s2)
    s2 = re.sub(r'(\S)@(\S*@)', r'\1@', s2)
    # remove double @ if ASR glitched
    s2 = s2.replace('@@', '@')
    return s2

# ---------------------------------------------------------
# NUMBERS
# ---------------------------------------------------------
NUM_WORD = {
    'zero': '0', 'oh': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
}

def words_to_digits(seq: List[str]) -> str:
    out = []
    i = 0
    while i < len(seq):
        tok = seq[i].lower()
        if tok in ('double', 'triple') and i+1 < len(seq):
            times = 2 if tok == 'double' else 3
            nxt = seq[i+1].lower()
            if nxt in NUM_WORD:
                out.append(NUM_WORD[nxt]*times)
                i += 2
                continue
        if tok in NUM_WORD:
            out.append(NUM_WORD[tok])
        i += 1
    return ''.join(out)

def normalize_numbers_spoken(s: str) -> str:
    tokens = s.split()
    result, buf = [], []
    for t in tokens:
        if t.lower() in NUM_WORD or t.lower() in ('double','triple'):
            buf.append(t)
        else:
            if buf:
                digits = words_to_digits(buf)
                result.append(digits)
                buf = []
            result.append(t)
    if buf:
        result.append(words_to_digits(buf))
    return ' '.join(result)

# ---------------------------------------------------------
# CURRENCY
# ---------------------------------------------------------
def normalize_currency(s: str) -> str:
    s = re.sub(r'\brupees?\b', '₹', s, flags=re.IGNORECASE)
    def group_indian(x):
        x = str(x)
        if len(x) <= 3: return x
        last3 = x[-3:]
        rest = x[:-3]
        return ','.join([rest[:-2*i or None][-2:] for i in range((len(rest)+1)//2)][::-1]) + ',' + last3
    def repl(m):
        num = re.sub(r'[^\d]', '', m.group(0))
        if num:
            return '₹' + group_indian(num)
        return m.group(0)
    return re.sub(r'₹\s*\d[\d,\.]*', repl, s)

# ---------------------------------------------------------
# NAME CORRECTION
# ---------------------------------------------------------
def correct_names_with_lexicon(s: str, names_lex: List[str], threshold: int = 85) -> str:
    """Correct misspelled Indian names using fuzzy matching.
       Skip tokens that look like emails or contain punctuation so we don't touch email usernames/domains.
    """
    tokens = s.split()
    out = []
    for i, t in enumerate(tokens):
        # Skip tokens that are clearly part of an email or contain punctuation/digits
        if '@' in t or '.' in t or re.search(r'[@\.\d]', t):
            out.append(t)
            continue

        # Keep capitalized first token (name) as-is if it looks like a proper name
        if i == 0 and t[0].isupper():
            out.append(t)
            continue

        best = process.extractOne(t, names_lex, scorer=fuzz.ratio)
        if best and best[1] >= threshold:
            # keep capitalization for names
            out.append(best[0].capitalize())
        else:
            out.append(t)
    return ' '.join(out)

# ---------------------------------------------------------
# FINAL CANDIDATE GENERATOR
# ---------------------------------------------------------
def generate_candidates(text: str, names_lex: List[str]) -> List[str]:
    cands = set()

    t = text.strip()
    t = re.sub(r'\s+', ' ', t)                   # fix spacing
    t = normalize_numbers_spoken(t)
    t = normalize_currency(t)
    t = normalize_email_tokens(t)
    t = correct_names_with_lexicon(t, names_lex)
    if not re.search(r'[.!?]$', t):
        if re.match(r'^(what|how|why|when|who|is|can|do|does|are)\b', t, re.I):
            t += '?'
        else:
            t += '.'
    cands.add(t)

    # Additional variants (safe for ranker)
    cands.add(normalize_email_tokens(text))
    cands.add(normalize_currency(text))
    cands.add(correct_names_with_lexicon(text, names_lex))
    cands.add(text)

    # deduplicate and limit to 4
    out = sorted(cands, key=len)[:4]
    return out
