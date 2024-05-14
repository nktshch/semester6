"""File for pytest fixtures."""

from unittest.mock import Mock

from _pytest.config import Config
from click.testing import CliRunner
import pytest
from pytest_mock import MockFixture


@pytest.fixture
def runner() -> CliRunner:
    """Runner."""
    return CliRunner()


@pytest.fixture
def mock_requests_get(mocker: MockFixture) -> Mock:
    """Mocker object to patch requests.get."""
    mock = mocker.patch("requests.get")
    mock.return_value.__enter__.return_value.json.return_value = {
        "title": "Lorem Ipsum",
        "extract": "Lorem ipsum dolor sit amet",
    }
    return mock


def pytest_configure(config: Config) -> None:
    """Configuration for pytests."""
    config.addinivalue_line("markers", "e2e: mark as end-to-end test.")
