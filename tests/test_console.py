"""Tests console.py."""

from unittest.mock import Mock

from click.testing import CliRunner
import pytest
from pytest_mock import MockFixture
import requests  # type: ignore

from semester6 import console


@pytest.fixture
def mock_wikipedia_random_page(mocker: MockFixture) -> Mock:
    return mocker.patch("semester6.wikipedia.random_page")


@pytest.mark.e2e
def test_main_succeeds_in_production_env(runner: CliRunner) -> None:
    """It exits with a status code of zero in production."""
    result = runner.invoke(console.main)
    assert result.exit_code == 0


def test_main_succeeds(runner: CliRunner, mock_requests_get: Mock) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(console.main)
    assert result.exit_code == 0


def test_main_prints_title(runner: CliRunner, mock_requests_get: Mock) -> None:
    """It prints sample string given by mock."""
    result = runner.invoke(console.main)
    assert "Lorem Ipsum" in result.output


def test_main_invokes_requests_get(runner: CliRunner, mock_requests_get: Mock) -> None:
    """It uses requests to retrieve page."""
    runner.invoke(console.main)
    assert mock_requests_get.called


def test_main_uses_ru_wikipedia_org(runner: CliRunner, mock_requests_get: Mock) -> None:
    """It uses Russian Wikipedia by default."""
    runner.invoke(console.main)
    args, _ = mock_requests_get.call_args
    assert "ru.wikipedia.org" in args[0]


def test_main_fails_on_request_error(
    runner: CliRunner, mock_requests_get: Mock
) -> None:
    """It fails if requests.get raises an exception."""
    mock_requests_get.side_effect = Exception("Sadness...")
    result = runner.invoke(console.main)
    assert result.exit_code == 1


def test_main_prints_message_on_request_error(
    runner: CliRunner, mock_requests_get: Mock
) -> None:
    """It prints message if requests raises an exception."""
    mock_requests_get.side_effect = requests.RequestException
    result = runner.invoke(console.main)
    assert "Error" in result.output


def test_main_uses_specified_language(
    runner: CliRunner, mock_wikipedia_random_page: Mock
) -> None:
    """It uses keyword argument."""
    runner.invoke(console.main, ["--language=pl"])
    mock_wikipedia_random_page.assert_called_with(language="pl")
