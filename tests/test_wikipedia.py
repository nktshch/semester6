"""Tests wikipedia.py."""

from unittest.mock import Mock

import click
import pytest

from semester6 import wikipedia


def test_random_page_uses_given_language(mock_requests_get: Mock) -> None:
    """It uses given language."""
    wikipedia.random_page(language="de")
    args, _ = mock_requests_get.call_args
    assert "de.wikipedia.org" in args[0]


def test_random_page_returns_page(mock_requests_get: Mock) -> None:
    """It returns dataclass object Page."""
    page = wikipedia.random_page()
    assert isinstance(page, wikipedia.Page)


def test_triggers_typeguard(mock_requests_get: Mock) -> None:
    """It does exactly that."""
    import json

    data = json.loads('{ "language": 1 }')
    wikipedia.random_page(language=data["language"])


def test_random_page_handles_validation_errors(mock_requests_get: Mock) -> None:
    """It raises exception if requests.get returns None."""
    mock_requests_get.return_value.__enter__.return_value.json.return_value = None
    with pytest.raises(click.ClickException):
        wikipedia.random_page()
