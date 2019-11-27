"""Metric logging."""


def log_scalar(name, step, value):
    """Logs a scalar."""
    # Format:
    #      1 | accuracy:                   0.789
    #   1234 | loss:                      12.345
    print('{:>6} | {:24}{:>9.3f}'.format(step, name + ':', value))
