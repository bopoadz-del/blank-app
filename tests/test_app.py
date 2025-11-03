"""Tests for the Streamlit application."""

import pytest
from unittest.mock import patch, MagicMock


def test_streamlit_import():
    """Test that streamlit can be imported."""
    import streamlit as st
    assert st is not None


def test_app_runs():
    """Test that the app file exists and can be imported."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("streamlit_app", "streamlit_app.py")
    assert spec is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules["streamlit_app"] = module

    # Mock streamlit functions to prevent actual app execution
    with patch('streamlit.title'), patch('streamlit.write'):
        spec.loader.exec_module(module)


@patch('streamlit.title')
@patch('streamlit.write')
def test_app_title(mock_write, mock_title):
    """Test that the app sets a title."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("streamlit_app", "streamlit_app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["streamlit_app"] = module
    spec.loader.exec_module(module)

    mock_title.assert_called_once()
    mock_write.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
