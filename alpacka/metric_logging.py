"""Metric logging."""


class StdoutLogger:
    """Logs to standard output."""

    @staticmethod
    def log_scalar(name, step, value):
        """Logs a scalar to stdout."""
        # Format:
        #      1 | accuracy:                   0.789
        #   1234 | loss:                      12.345
        print('{:>6} | {:24}{:>9.3f}'.format(step, name + ':', value))


_loggers = [StdoutLogger]


def register_logger(logger):
    """Adds a logger to log to."""
    _loggers.append(logger)


def log_scalar(name, value):
    """Logs a scalar to the loggers."""
    for logger in _loggers:
        logger.log_scalar(name, _epoch, value)


_epoch = 0


def get_global_epoch():
    return _epoch


def inc_global_epoch():
    global _epoch  # pylint: disable=global-statement
    _epoch += 1
