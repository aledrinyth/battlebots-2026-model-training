"""Reconstructed teaching example, not recovered historical source.
See notes/engine-upgrade-v2026.1.11.md, Blocker 2. Run this file directly.
"""
def visible(x, y, radius, half_window):
    # BUG: a circle may intersect the view even when its center lies outside.
    return abs(x) <= half_window and abs(y) <= half_window

if __name__ == '__main__':
    result = visible(10.5, 0.0, 1.0, 10.0)
    assert result is False
    print('REPRODUCED: point-only test hides an intersecting circle; expected True, got False')
