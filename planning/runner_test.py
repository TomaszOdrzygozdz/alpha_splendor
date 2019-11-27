"""Tests for planning.runner."""


from planning import runner


def test_smoke(capsys):
    n_epochs = 3
    runner.Runner(
        n_envs=2,
        n_epochs=n_epochs,
    ).run()

    # Check that metrics were printed in each epoch.
    captured = capsys.readouterr()
    assert captured.out.count('return_mean') == n_epochs
