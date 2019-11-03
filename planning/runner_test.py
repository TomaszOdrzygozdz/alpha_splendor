"""Tests for planning.runner."""


from planning import runner


def test_smoke():
    runner.Runner(
        n_envs=2,
        n_epochs=3,
    ).run()
