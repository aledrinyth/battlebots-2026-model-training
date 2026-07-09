"""Reconstructed correction of the row-order assertion (small exact example).
Production uses tolerant nearest-row matching; see notes/P3.4-logging-checkpointing.md section 5.
"""
from collections import Counter

def equivalent(a, b):
    # Preserve row multiplicity while ignoring order. A set would lose duplicates.
    return Counter(a) == Counter(b)

if __name__ == '__main__':
    a = [(1.0, 0.0), (0.0, 1.0)]
    assert equivalent(a, list(reversed(a)))
    assert not equivalent(a, [(1.0, 0.0), (1.0, 0.0)])
    print('FIX VERIFIED: permutations accepted; different multiplicities rejected')
