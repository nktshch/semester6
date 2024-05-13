"""File for pytest fixtures"""

from unittest.mock import Mock

from _pytest.config import Config
import click
from click.testing import CliRunner
import pytest
from pytest_mock import MockFixture


@pytest.fixture
def runner() -> CliRunner:
    return click.testing.CliRunner()


@pytest.fixture
def mock_requests_get(mocker: MockFixture) -> Mock:
    mock = mocker.patch("requests.get")
    mock.return_value.__enter__.return_value.json.return_value = {
        "title": "Lorem Ipsum",
        "extract": "Lorem ipsum dolor sit amet",
    }
    return mock


def pytest_configure(config: Config) -> None:
    config.addinivalue_line("markers", "e2e: mark as end-to-end test.")
