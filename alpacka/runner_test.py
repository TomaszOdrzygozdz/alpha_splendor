"""Tests for alpacka.runner."""


from alpacka import runner
from alpacka import metric_logging


def test_smoke(tmpdir, capsys):
    n_epochs = 3
    runner.Runner(
        output_dir=tmpdir,
        n_envs=2,
        n_epochs=n_epochs,
        log_fns=[metric_logging.log_scalar]
    ).run()

    # Check that metrics were printed in each epoch.
    captured = capsys.readouterr()
    assert captured.out.count('return_mean') == n_epochs
