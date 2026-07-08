"""Reconstructed example of the documented overly strict test assertion.
See notes/P3.4-logging-checkpointing.md section 5. No production bug implied.
"""
def equivalent(a, b):
    # BUG in the test: harmless reordering of equidistant entities fails.
    return a == b

if __name__ == '__main__':
    a = [(1.0, 0.0), (0.0, 1.0)]
    assert not equivalent(a, list(reversed(a)))
    print('REPRODUCED: elementwise comparison rejects the same two rows in different order')
