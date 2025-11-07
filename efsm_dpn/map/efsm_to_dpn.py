"""
EFSM to Data-Aware Petri Net mapping.

Translates EFSM models to pm4py Petri nets with data annotations.

:return : EFSM to DPN mapping utilities.
:return: Functions for translating EFSM to DPN.
"""

from typing import Dict
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from efsm_dpn.models.efsm import EFSM
from efsm_dpn.models.dpn import DPN, DPNTransition
from efsm_dpn.learn.guard_inference import infer_read_write_sets


def map_efsm_to_dpn(efsm: EFSM) -> DPN:
    """
    Map an EFSM to a Data-Aware Petri Net.

    :param efsm: Extended Finite State Machine.
    :return : Data-Aware Petri Net.
    :return: DPN representation of EFSM.
    """
    net = PetriNet(name="EFSM_DPN")

    place_map: Dict[str, PetriNet.Place] = {}
    for state_name in efsm.states:
        place = PetriNet.Place(state_name)
        net.places.add(place)
        place_map[state_name] = place

    initial_marking = Marking()
    initial_place = place_map[efsm.initial]
    initial_marking[initial_place] = 1

    data_transitions: Dict[str, DPNTransition] = {}

    for idx, trans in enumerate(efsm.transitions):
        trans_name = f"t{idx}_{trans.label}"
        pn_trans = PetriNet.Transition(name=trans_name, label=trans.label)
        net.transitions.add(pn_trans)

        source_place = place_map[trans.source]
        target_place = place_map[trans.target]

        petri_utils.add_arc_from_to(source_place, pn_trans, net)
        petri_utils.add_arc_from_to(pn_trans, target_place, net)

        read_vars, write_vars = infer_read_write_sets(
            trans.guard, trans.update.assignments
        )

        dpn_trans = DPNTransition(
            pn_transition=pn_trans,
            guard=trans.guard,
            update=trans.update,
            read_vars=read_vars,
            write_vars=write_vars,
        )
        data_transitions[trans_name] = dpn_trans

    variables = {var.name: var.dtype for var in efsm.variables.values()}

    dpn = DPN(
        petri_net=net,
        initial_marking=initial_marking,
        final_marking=None,
        data_transitions=data_transitions,
        variables=variables,
    )

    return dpn
