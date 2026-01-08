def clamp11(x):
    """Clamp a value to [-1.0, 1.0].

    Args:
        x: Input value.

    Returns:
        Clamped value in [-1.0, 1.0].
    """
    if x < -1.0:
        return -1.0
    if x > 1.0:
        return 1.0
    return x


def clip(x, lo, hi):
    """Clamp a value to [lo, hi].

    Args:
        x: Input value.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped value.
    """
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
