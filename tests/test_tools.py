"""Tests for tools: calculator and arxiv_search."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from app.tools.calculator import calculator
from app.tools.registry import TOOL_REGISTRY


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class TestCalculator:
    def test_addition(self):
        assert calculator("2 + 3") == pytest.approx(5.0)

    def test_subtraction(self):
        assert calculator("10 - 4") == pytest.approx(6.0)

    def test_multiplication(self):
        assert calculator("3 * 7") == pytest.approx(21.0)

    def test_division(self):
        assert calculator("10 / 4") == pytest.approx(2.5)

    def test_power(self):
        assert calculator("2 ** 10") == pytest.approx(1024.0)

    def test_nested_parentheses(self):
        assert calculator("(2 + 3) * (4 - 1)") == pytest.approx(15.0)

    def test_unary_minus(self):
        assert calculator("-5") == pytest.approx(-5.0)

    def test_float(self):
        assert calculator("1.5 + 2.5") == pytest.approx(4.0)

    def test_complex_expression(self):
        assert calculator("(12 + 8) * 3") == pytest.approx(60.0)

    def test_rejects_function_calls(self):
        with pytest.raises((ValueError, Exception)):
            calculator("__import__('os').system('ls')")

    def test_rejects_attribute_access(self):
        with pytest.raises((ValueError, Exception)):
            calculator("(1).bit_length()")

    def test_rejects_names(self):
        with pytest.raises((ValueError, Exception)):
            calculator("x + 1")

    def test_division_by_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            calculator("1 / 0")


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_calculator_registered(self):
        assert "calculator" in TOOL_REGISTRY

    def test_arxiv_search_registered(self):
        assert "arxiv_search" in TOOL_REGISTRY

    def test_calculator_callable(self):
        result = TOOL_REGISTRY["calculator"](expression="2+2")
        assert result == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# arXiv search (mocked network)
# ---------------------------------------------------------------------------

class TestArxivTool:
    _FAKE_ATOM = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
    <entry>
      <id>http://arxiv.org/abs/2401.00001v1</id>
      <title>Test Paper on Attention</title>
      <summary>This paper discusses attention mechanisms.</summary>
      <link title="pdf" href="https://arxiv.org/pdf/2401.00001.pdf" />
    </entry>
    </feed>"""

    def _make_mock_resp(self):
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.text = self._FAKE_ATOM
        return mock

    def test_returns_list_of_dicts(self):
        with patch("requests.get", return_value=self._make_mock_resp()):
            results = TOOL_REGISTRY["arxiv_search"](query="attention")
        assert isinstance(results, list)

    def test_result_has_expected_fields(self):
        with patch("requests.get", return_value=self._make_mock_resp()):
            results = TOOL_REGISTRY["arxiv_search"](query="attention")
        if results:
            r = results[0]
            assert "title" in r
            assert "summary" in r
            assert "pdf_url" in r

    def test_returns_empty_on_no_entries(self):
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.text = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        with patch("requests.get", return_value=mock):
            results = TOOL_REGISTRY["arxiv_search"](query="nothing")
        assert results == []

    def test_network_error_propagates(self):
        with patch("requests.get", side_effect=Exception("network error")):
            with pytest.raises(Exception):
                TOOL_REGISTRY["arxiv_search"](query="test")
