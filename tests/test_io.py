"""
Unit tests for log I/O functionality.

:return : Test suite.
:return: Unit tests for log reading and processing.
"""

import pytest
import pandas as pd
from pathlib import Path
from efsm_dpn.logs.io import (
    read_log,
    extract_traces,
    infer_attribute_domains,
    detect_variable_propagation,
)


def test_extract_traces() -> None:
    """
    Test trace extraction from DataFrame.

    :return : None.
    :return: Test assertion.
    """
    df = pd.DataFrame(
        {
            "case_id": ["c1", "c1", "c2", "c2"],
            "activity": ["A", "B", "A", "C"],
            "timestamp": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]
            ),
            "amount": [100, 100, 50, 50],
        }
    )

    traces = extract_traces(df)

    assert len(traces) == 2
    assert len(traces[0]) == 2
    assert traces[0][0][0] == "A"
    assert traces[0][1][0] == "B"
    assert traces[0][0][1]["amount"] == 100


def test_infer_attribute_domains() -> None:
    """
    Test attribute domain inference.

    :return : None.
    :return: Test assertion.
    """
    df = pd.DataFrame(
        {
            "case_id": ["c1", "c1"],
            "activity": ["A", "B"],
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "amount": [100, 200],
            "category": ["urgent", "normal"],
        }
    )

    domains = infer_attribute_domains(df)

    assert "amount" in domains
    assert domains["amount"]["dtype"] == "int"
    assert domains["amount"]["min"] == 100
    assert domains["amount"]["max"] == 200

    assert "category" in domains
    assert domains["category"]["dtype"] == "cat"
    assert set(domains["category"]["values"]) == {"urgent", "normal"}


def test_detect_variable_propagation() -> None:
    """
    Test variable propagation detection.

    :return : None.
    :return: Test assertion.
    """
    traces = [
        [("A", {"x": 10}), ("B", {"x": 10}), ("C", {"x": 10})],
        [("A", {"x": 20}), ("B", {"x": 20}), ("C", {"x": 20})],
    ]

    propagation = detect_variable_propagation(traces)

    assert "x" in propagation
    assert propagation["x"] == "persistent"
