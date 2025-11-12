"""
EFSM to Data-Aware Petri Net mapping.

Translates EFSM models to pm4py Petri nets with data annotations.

:return : EFSM to DPN mapping utilities.
:return: Functions for translating EFSM to DPN.
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from efsm_dpn.models.efsm import EFSM
from efsm_dpn.models.dpn import DPN, DPNTransition
from efsm_dpn.learn.guard_inference import infer_read_write_sets


def map_efsm_to_dpn(efsm: EFSM) -> DPN:
    """
    Map an EFSM to a Data-Aware Petri Net.
    
    Creates a simplified DPN with one transition per activity label.
    Uses a minimal control flow structure with shared places for sequential flow.

    :param efsm: Extended Finite State Machine.
    :return : Data-Aware Petri Net.
    :return: DPN representation of EFSM.
    """
    net = PetriNet(name="EFSM_DPN")
    data_transitions: Dict[str, DPNTransition] = {}

    # Group transitions by label
    transitions_by_label: Dict[str, List] = defaultdict(list)
    for trans in efsm.transitions:
        transitions_by_label[trans.label].append(trans)

    # Create start and end places
    start_place = PetriNet.Place("start")
    net.places.add(start_place)
    initial_marking = Marking()
    initial_marking[start_place] = 1
    
    end_place = PetriNet.Place("end")
    net.places.add(end_place)
    
    # Create one intermediate place (represents "in process" state)
    # All activities will consume from and produce to this place
    process_place = PetriNet.Place("process")
    net.places.add(process_place)
    
    # Connect start to process with a silent transition
    start_trans = PetriNet.Transition(name="start_process", label=None)
    net.transitions.add(start_trans)
    petri_utils.add_arc_from_to(start_place, start_trans, net)
    petri_utils.add_arc_from_to(start_trans, process_place, net)
    
    # Create one visible transition per activity
    # Each consumes from and produces to the process place (self-loop)
    for label, trans_group in sorted(transitions_by_label.items()):
        trans_name = f"{label}"
        pn_trans = PetriNet.Transition(name=trans_name, label=label)
        net.transitions.add(pn_trans)
        
        # Self-loop: process -> activity -> process
        petri_utils.add_arc_from_to(process_place, pn_trans, net)
        petri_utils.add_arc_from_to(pn_trans, process_place, net)
        
        # Merge guards from all EFSM transitions with this label
        from efsm_dpn.models.efsm import Guard, Update
        
        guard_strings = []
        for trans in trans_group:
            if trans.guard and trans.guard.serialized and trans.guard.serialized != "true":
                guard_strings.append(f"({trans.guard.serialized})")
        
        if guard_strings:
            if len(guard_strings) == 1:
                merged_guard_str = guard_strings[0]
            else:
                merged_guard_str = " Or ".join(guard_strings)
            merged_guard = Guard(expression=None, serialized=merged_guard_str)
        else:
            merged_guard = Guard(expression=None, serialized="true")
        
        # Merge all update assignments
        merged_assignments = {}
        for trans in trans_group:
            merged_assignments.update(trans.update.assignments)
        merged_update = Update(assignments=merged_assignments)
        
        read_vars, write_vars = infer_read_write_sets(
            merged_guard, merged_assignments
        )

        dpn_trans = DPNTransition(
            pn_transition=pn_trans,
            guard=merged_guard,
            update=merged_update,
            read_vars=read_vars,
            write_vars=write_vars,
        )
        data_transitions[trans_name] = dpn_trans
    
    # Connect process to end with a silent transition
    end_trans = PetriNet.Transition(name="end_process", label=None)
    net.transitions.add(end_trans)
    petri_utils.add_arc_from_to(process_place, end_trans, net)
    petri_utils.add_arc_from_to(end_trans, end_place, net)

    variables = {var.name: var.dtype for var in efsm.variables.values()}

    dpn = DPN(
        petri_net=net,
        initial_marking=initial_marking,
        final_marking=None,
        data_transitions=data_transitions,
        variables=variables,
    )

    return dpn

