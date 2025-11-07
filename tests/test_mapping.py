"""
Unit tests for EFSM to DPN mapping.

:return : Test suite.
:return: Unit tests for mapping functionality.
"""

import pytest
from efsm_dpn.models.efsm import EFSM, Variable, Guard, Update, Transition
from efsm_dpn.map.efsm_to_dpn import map_efsm_to_dpn


def test_efsm_to_dpn_mapping() -> None:
    """
    Test EFSM to DPN conversion.

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

    dpn = map_efsm_to_dpn(efsm)

    assert len(dpn.petri_net.places) == 3
    assert len(dpn.petri_net.transitions) == 2
    assert len(dpn.initial_marking) == 1

    initial_places = [p for p, count in dpn.initial_marking.items() if count > 0]
    assert len(initial_places) == 1
    assert initial_places[0].name == "s0"


def test_dpn_data_annotations() -> None:
    """
    Test DPN data annotations preservation.

    :return : None.
    :return: Test assertion.
    """
    variables = {"x": Variable(name="x", dtype="int")}
    states = {"s0", "s1"}
    transitions = [
        Transition(
            source="s0",
            label="A",
            guard=Guard(expression=None, serialized="x >= 100"),
            update=Update(assignments={"x": "attr.x"}),
            target="s1",
        )
    ]

    efsm = EFSM(states=states, initial="s0", variables=variables, transitions=transitions)

    dpn = map_efsm_to_dpn(efsm)

    assert len(dpn.data_transitions) == 1

    dpn_trans = list(dpn.data_transitions.values())[0]
    assert dpn_trans.guard.serialized == "x >= 100"
    assert "x" in dpn_trans.update.assignments
