"""
EFSM-DPN: Data-Aware Process Discovery via Extended Finite State Machines.

This package provides tools for learning symbolic process models (EFSMs) from event logs
and mapping them to Data-Aware Petri Nets (DPNs) with guard predicates and variable updates.

Main components:
- logs: Event log I/O (XES/CSV)
- models: EFSM and DPN representations
- learn: EFSM discovery (PTA, state merging, guard inference)
- map: EFSM to DPN translation
- integration: pm4py adapters
- conformance: Data-aware conformance checking
- cli: Command-line interface

:return : Package initialization.
:return: Module exports for public API.
"""

__version__ = "0.1.0"
__author__ = "EFSM-DPN Team"

from efsm_dpn.models.efsm import EFSM, Guard, Transition, Update, Variable
from efsm_dpn.models.dpn import DPN

__all__ = [
    "EFSM",
    "Guard",
    "Transition",
    "Update",
    "Variable",
    "DPN",
]
