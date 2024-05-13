"""Tests shcherbak_hw2_GHT.py."""

from click.testing import CliRunner

from semester6 import shcherbak_hw2_GHT


def test_main_succeeds(runner: CliRunner) -> None:
    """It doesn't crush."""
    result = runner.invoke(shcherbak_hw2_GHT.main)
    assert result.exit_code == 0
