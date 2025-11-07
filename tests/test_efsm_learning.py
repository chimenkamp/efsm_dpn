"""
Unit tests for EFSM learning.

:return : Test suite.
:return: Unit tests for EFSM discovery.
"""

import pytest
from efsm_dpn.learn.pta import build_pta
from efsm_dpn.learn.state_merger import are_states_compatible
from efsm_dpn.models.efsm import EFSM, Variable, Guard, Update, Transition


def test_build_pta() -> None:
    """
    Test PTA construction.

    :return : None.
    :return: Test assertion.
    """
    traces = [
        [("A", {"x": 10}), ("B", {"x": 10})],
        [("A", {"x": 20}), ("C", {"x": 20})],
    ]

    pta = build_pta(traces)

    assert pta.root is not None
    assert len(pta.root.children) == 1
    assert "A" in pta.root.children


def test_efsm_simulation() -> None:
    """
    Test EFSM trace simulation.

    :return : None.
    :return: Test assertion.
    """
    variables = {"x": Variable(name="x", dtype="int")}
    states = {"s0", "s1", "s2"}
    transitions = [
        Transition(
            source="s0",
            label="A",
            guard=Guard(expression=None, serialized="true"),
            update=Update(assignments={"x": "attr.x"}),
            target="s1",
        ),
        Transition(
            source="s1",
            label="B",
            guard=Guard(expression=None, serialized="true"),
            update=Update(assignments={}),
            target="s2",
        ),
    ]

    efsm = EFSM(states=states, initial="s0", variables=variables, transitions=transitions)

    trace = [("A", {"x": 10}), ("B", {})]
    accepted, path, var_state = efsm.simulate_trace(trace)

    assert accepted is True
    assert len(path) == 3
    assert path == ["s0", "s1", "s2"]
    assert var_state["x"] == 10


def test_efsm_serialization() -> None:
    """
    Test EFSM JSON serialization.

    :return : None.
    :return: Test assertion.
    """
    variables = {"x": Variable(name="x", dtype="int")}
    states = {"s0", "s1"}
    transitions = [
        Transition(
            source="s0",
            label="A",
            guard=Guard(expression=None, serialized="true"),
            update=Update(assignments={}),
            target="s1",
        )
    ]

    efsm = EFSM(states=states, initial="s0", variables=variables, transitions=transitions)

    efsm_dict = efsm.to_dict()
    efsm_restored = EFSM.from_dict(efsm_dict)

    assert efsm_restored.states == efsm.states
    assert efsm_restored.initial == efsm.initial
    assert len(efsm_restored.transitions) == len(efsm.transitions)
