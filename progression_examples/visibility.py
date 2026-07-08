"""Reconstructed correction of the point-only visibility example.
Food uses a point test; blobs/viruses need a circle-box intersection test.
See notes/engine-upgrade-v2026.1.11.md, Blocker 2. Run this file directly.
"""
def visible(x, y, radius, half_window):
    # Distance from the circle center to the closest point on the view box.
    dx = max(abs(x) - half_window, 0.0)
    dy = max(abs(y) - half_window, 0.0)
    return dx * dx + dy * dy <= radius * radius

if __name__ == '__main__':
    assert visible(10.5, 0.0, 1.0, 10.0)
    assert not visible(12.0, 0.0, 1.0, 10.0)
    assert visible(11.0, 0.0, 1.0, 10.0)
    print('FIX VERIFIED: circle intersection, outside case, and boundary case')
