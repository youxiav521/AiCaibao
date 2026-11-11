import sys
sys.stdout.reconfigure(encoding='utf-8')
from difflib import SequenceMatcher

def token_sort_similarity(a, b):
    a_sorted = ' '.join(sorted(a.split()))
    b_sorted = ' '.join(sorted(b.split()))
    return SequenceMatcher(None, a_sorted, b_sorted).ratio()

pairs = [
    ('cash and bank balances', 'bank and cash balances'),
    ('cash and bank balances', 'cash and bank balances'),
    ('receivables and payables', 'payables and receivables'),
    ('loans and advances to banks', 'advances and loans to banks'),
    ('property plant and equipment', 'plant and property and equipment'),
    ('current assets', 'non-current assets'),
    ('operating income', 'operating expense'),
    ('tangible and intangible assets', 'intangible and tangible assets'),
]

print(f'{"Query":<45} {"Candidate":<45} {"Original":>8} {"TokenSort":>8}')
print('-' * 110)
for q, c in pairs:
    orig = SequenceMatcher(None, q, c).ratio()
    ts = token_sort_similarity(q, c)
    match_orig = 'YES' if orig >= 0.9 else 'NO'
    match_ts = 'YES' if ts >= 0.9 else 'NO'
    print(f'{q:<45} {c:<45} {orig:>5.3f}({match_orig:>3}) {ts:>5.3f}({match_ts:>3})')
