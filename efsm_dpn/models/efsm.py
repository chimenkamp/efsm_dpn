"""
Extended Finite State Machine (EFSM) model definitions.

This module provides dataclasses and utilities for representing EFSMs:
- EFSM = ⟨S, s0, X, Σ, T⟩
- S: finite set of control states
- s0: initial state
- X: finite set of typed variables
- Σ: alphabet of event labels (activity names)
- T: transitions with guards and updates

:return : EFSM model components.
:return: Classes for Variable, Guard, Update, Transition, and EFSM.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
import json
from z3 import *


@dataclass
class Variable:
    """
    Typed variable in an EFSM.

    :param name: Variable name.
    :param dtype: Data type (int, float, string, cat).
    :return : Variable instance.
    :return: A typed variable descriptor.
    """

    name: str
    dtype: Literal["int", "float", "string", "cat"]

    def to_dict(self) -> Dict[str, str]:
        """
        Serialize variable to dictionary.

        :return : Dictionary representation.
        :return: Dict with name and dtype.
        """
        return {"name": self.name, "dtype": self.dtype}

    @staticmethod
    def from_dict(data: Dict[str, str]) -> "Variable":
        """
        Deserialize variable from dictionary.

        :param data: Dictionary with name and dtype.
        :return : Variable instance.
        :return: Reconstructed Variable.
        """
        return Variable(name=data["name"], dtype=data["dtype"])


@dataclass
class Guard:
    """
    Predicate guard for an EFSM transition.

    Wraps a z3 expression and provides serialization.

    :param expression: Z3 boolean expression or None (true guard).
    :param serialized: JSON-serializable representation for persistence.
    :return : Guard instance.
    :return: A guard predicate.
    """

    expression: Optional[z3.ExprRef] = None
    serialized: Optional[str] = None

    def evaluate(self, var_state: Dict[str, Any]) -> bool:
        """
        Evaluate guard against variable state.

        :param var_state: Dictionary mapping variable names to values.
        :return : Boolean result.
        :return: True if guard satisfied, False otherwise.
        """
        if self.expression is None:
            return True
        try:
            solver = z3.Solver()
            substituted = self.expression
            for var_name, value in var_state.items():
                z3_var = z3.Int(var_name) if isinstance(value, int) else z3.Real(var_name)
                if isinstance(value, (int, float)):
                    substituted = z3.substitute(substituted, (z3_var, z3.RealVal(value) if isinstance(value, float) else z3.IntVal(value)))
            solver.add(substituted)
            return solver.check() == z3.sat
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Optional[str]]:
        """
        Serialize guard to dictionary.

        :return : Dictionary representation.
        :return: Dict with serialized expression.
        """
        if self.expression is None:
            return {"serialized": None}
        if self.serialized is None:
            self.serialized = str(self.expression)
        return {"serialized": self.serialized}

    @staticmethod
    def from_dict(data: Dict[str, Optional[str]]) -> "Guard":
        """
        Deserialize guard from dictionary.

        :param data: Dictionary with serialized expression.
        :return : Guard instance.
        :return: Reconstructed Guard.
        """
        serialized = data.get("serialized")
        if serialized is None:
            return Guard(expression=None, serialized=None)
        return Guard(expression=None, serialized=serialized)

    def __str__(self) -> str:
        """
        String representation of guard.

        :return : String form.
        :return: Guard expression as string.
        """
        if self.expression is None and self.serialized is None:
            return "true"
        return self.serialized if self.serialized else str(self.expression)


@dataclass
class Update:
    """
    Variable update map for an EFSM transition.

    Maps variable names to assignment expressions.

    :param assignments: Dictionary mapping variable names to expressions.
    :return : Update instance.
    :return: A variable update descriptor.
    """

    assignments: Dict[str, str] = field(default_factory=dict)

    def apply(self, var_state: Dict[str, Any], event_attrs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply updates to variable state.

        :param var_state: Current variable state.
        :param event_attrs: Event attributes available for assignment.
        :return : Updated variable state.
        :return: New state after applying assignments.
        """
        new_state = var_state.copy()
        for var_name, expr in self.assignments.items():
            try:
                if expr.startswith("attr."):
                    attr_name = expr[5:]
                    if attr_name in event_attrs:
                        new_state[var_name] = event_attrs[attr_name]
                else:
                    local_vars = {**var_state, **event_attrs}
                    new_state[var_name] = eval(expr, {"__builtins__": {}}, local_vars)
            except Exception:
                pass
        return new_state

    def to_dict(self) -> Dict[str, Dict[str, str]]:
        """
        Serialize update to dictionary.

        :return : Dictionary representation.
        :return: Dict with assignments.
        """
        return {"assignments": self.assignments}

    @staticmethod
    def from_dict(data: Dict[str, Dict[str, str]]) -> "Update":
        """
        Deserialize update from dictionary.

        :param data: Dictionary with assignments.
        :return : Update instance.
        :return: Reconstructed Update.
        """
        return Update(assignments=data.get("assignments", {}))

    def __str__(self) -> str:
        """
        String representation of update.

        :return : String form.
        :return: Assignments as string.
        """
        if not self.assignments:
            return "ε"
        items = [f"{k} := {v}" for k, v in self.assignments.items()]
        return "; ".join(items)


@dataclass
class Transition:
    """
    EFSM transition.

    :param source: Source state name.
    :param label: Activity label (event name).
    :param guard: Guard predicate.
    :param update: Variable updates.
    :param target: Target state name.
    :return : Transition instance.
    :return: An EFSM transition.
    """

    source: str
    label: str
    guard: Guard
    update: Update
    target: str

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize transition to dictionary.

        :return : Dictionary representation.
        :return: Dict with all transition components.
        """
        return {
            "source": self.source,
            "label": self.label,
            "guard": self.guard.to_dict(),
            "update": self.update.to_dict(),
            "target": self.target,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Transition":
        """
        Deserialize transition from dictionary.

        :param data: Dictionary with transition data.
        :return : Transition instance.
        :return: Reconstructed Transition.
        """
        return Transition(
            source=data["source"],
            label=data["label"],
            guard=Guard.from_dict(data["guard"]),
            update=Update.from_dict(data["update"]),
            target=data["target"],
        )

    def __str__(self) -> str:
        """
        String representation of transition.

        :return : String form.
        :return: Transition as string.
        """
        return f"{self.source} --[{self.label}]/{self.guard}/{self.update}--> {self.target}"


@dataclass
class EFSM:
    """
    Extended Finite State Machine.

    :param states: Set of state names.
    :param initial: Initial state name.
    :param variables: Dictionary of variables by name.
    :param transitions: List of transitions.
    :return : EFSM instance.
    :return: An EFSM model.
    """

    states: Set[str]
    initial: str
    variables: Dict[str, Variable]
    transitions: List[Transition] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Validate EFSM consistency.

        :return : None.
        :return: Validation side-effect.
        """
        if self.initial not in self.states:
            raise ValueError(f"Initial state {self.initial} not in states")
        for t in self.transitions:
            if t.source not in self.states:
                raise ValueError(f"Transition source {t.source} not in states")
            if t.target not in self.states:
                raise ValueError(f"Transition target {t.target} not in states")

    def simulate_trace(
        self, events: List[Tuple[str, Dict[str, Any]]]
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Simulate EFSM on a trace of events.

        :param events: List of (activity, attributes) tuples.
        :return : Tuple (accepted, state_path, final_var_state).
        :return: Simulation result with acceptance status and path.
        """
        current_state = self.initial
        var_state: Dict[str, Any] = {v.name: None for v in self.variables.values()}
        state_path = [current_state]

        for activity, attrs in events:
            candidates = [
                t
                for t in self.transitions
                if t.source == current_state and t.label == activity
            ]
            fired = False
            for t in candidates:
                if t.guard.evaluate(var_state):
                    var_state = t.update.apply(var_state, attrs)
                    current_state = t.target
                    state_path.append(current_state)
                    fired = True
                    break
            if not fired:
                return False, state_path, var_state

        return True, state_path, var_state

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize EFSM to dictionary.

        :return : Dictionary representation.
        :return: Dict with all EFSM components.
        """
        return {
            "states": list(self.states),
            "initial": self.initial,
            "variables": {name: var.to_dict() for name, var in self.variables.items()},
            "transitions": [t.to_dict() for t in self.transitions],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EFSM":
        """
        Deserialize EFSM from dictionary.

        :param data: Dictionary with EFSM data.
        :return : EFSM instance.
        :return: Reconstructed EFSM.
        """
        return EFSM(
            states=set(data["states"]),
            initial=data["initial"],
            variables={
                name: Variable.from_dict(vdata)
                for name, vdata in data["variables"].items()
            },
            transitions=[Transition.from_dict(t) for t in data["transitions"]],
        )

    def to_json(self, filepath: str) -> None:
        """
        Save EFSM to JSON file.

        :param filepath: Path to output file.
        :return : None.
        :return: File write side-effect.
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def from_json(filepath: str) -> "EFSM":
        """
        Load EFSM from JSON file.

        :param filepath: Path to input file.
        :return : EFSM instance.
        :return: Loaded EFSM.
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return EFSM.from_dict(data)
