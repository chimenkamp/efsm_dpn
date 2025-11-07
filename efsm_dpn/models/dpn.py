"""
Data-Aware Petri Net (DPN) model definitions.

This module provides a wrapper around pm4py Petri nets with data annotations.
Each transition carries guard predicates and read/write variable sets.

:return : DPN model components.
:return: Classes for DPN representation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from pm4py.objects.petri_net.obj import PetriNet, Marking
from efsm_dpn.models.efsm import Guard, Update


@dataclass
class DPNTransition:
    """
    Data-aware Petri net transition.

    :param pn_transition: Underlying pm4py transition object.
    :param guard: Guard predicate.
    :param update: Variable updates.
    :param read_vars: Variables read by guard.
    :param write_vars: Variables written by update.
    :return : DPNTransition instance.
    :return: A data-aware transition.
    """

    pn_transition: PetriNet.Transition
    guard: Guard
    update: Update
    read_vars: Set[str] = field(default_factory=set)
    write_vars: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize DPN transition to dictionary.

        :return : Dictionary representation.
        :return: Dict with transition data.
        """
        return {
            "name": self.pn_transition.name,
            "label": self.pn_transition.label,
            "guard": self.guard.to_dict(),
            "update": self.update.to_dict(),
            "read_vars": list(self.read_vars),
            "write_vars": list(self.write_vars),
        }


@dataclass
class DPN:
    """
    Data-Aware Petri Net.

    Wraps a pm4py PetriNet with data annotations on transitions.

    :param petri_net: Underlying pm4py Petri net.
    :param initial_marking: Initial marking.
    :param final_marking: Final marking (optional).
    :param data_transitions: Map from transition names to DPNTransition.
    :param variables: Dictionary of variable names to types.
    :return : DPN instance.
    :return: A data-aware Petri net model.
    """

    petri_net: PetriNet
    initial_marking: Marking
    final_marking: Optional[Marking] = None
    data_transitions: Dict[str, DPNTransition] = field(default_factory=dict)
    variables: Dict[str, str] = field(default_factory=dict)

    def get_transition_guard(self, transition_name: str) -> Optional[Guard]:
        """
        Get guard for a transition.

        :param transition_name: Name of transition.
        :return : Guard or None.
        :return: Guard predicate if exists.
        """
        dt = self.data_transitions.get(transition_name)
        return dt.guard if dt else None

    def get_transition_update(self, transition_name: str) -> Optional[Update]:
        """
        Get update for a transition.

        :param transition_name: Name of transition.
        :return : Update or None.
        :return: Update assignments if exist.
        """
        dt = self.data_transitions.get(transition_name)
        return dt.update if dt else None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize DPN to dictionary.

        :return : Dictionary representation.
        :return: Dict with DPN components.
        """
        return {
            "places": [p.name for p in self.petri_net.places],
            "transitions": [
                self.data_transitions[t.name].to_dict()
                if t.name in self.data_transitions
                else {"name": t.name, "label": t.label}
                for t in self.petri_net.transitions
            ],
            "arcs": [
                {
                    "source": arc.source.name,
                    "target": arc.target.name,
                    "weight": arc.weight,
                }
                for arc in self.petri_net.arcs
            ],
            "initial_marking": {p.name: count for p, count in self.initial_marking.items()},
            "final_marking": (
                {p.name: count for p, count in self.final_marking.items()}
                if self.final_marking
                else None
            ),
            "variables": self.variables,
        }
