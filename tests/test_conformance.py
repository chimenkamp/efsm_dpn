"""
Unit tests for conformance checking.

:return : Test suite.
:return: Unit tests for conformance evaluation.
"""

import pytest
from efsm_dpn.models.efsm import EFSM, Variable, Guard, Update, Transition
from efsm_dpn.map.efsm_to_dpn import map_efsm_to_dpn
from efsm_dpn.conformance.checks import evaluate_guard_satisfaction


def test_guard_evaluation() -> None:
    """
    Test guard evaluation on traces.

    :return : None.
    :return: Test assertion.
    """
    guard = Guard(expression=None, serialized="true")
    var_state = {"x": 100}

    result = guard.evaluate(var_state)
    assert result is True


def test_update_application() -> None:
    """
    Test variable update application.

    :return : None.
    :return: Test assertion.
    """
    update = Update(assignments={"x": "attr.amount", "y": "x + 10"})
    var_state = {"x": 50, "y": 0}
    event_attrs = {"amount": 100}

    new_state = update.apply(var_state, event_attrs)

    assert new_state["x"] == 100
    assert new_state["y"] == 60
