"""
EFSM learner orchestration.

Main pipeline for learning EFSMs from event logs.

:return : EFSM learning pipeline.
:return: Functions for end-to-end EFSM discovery.
"""

from typing import Any, Dict, List, Optional, Tuple
from efsm_dpn.logs.io import (
    read_log,
    extract_traces,
    infer_attribute_domains,
    detect_variable_propagation,
)
from efsm_dpn.learn.pta import build_pta, PTA, PTANode
from efsm_dpn.learn.state_merger import blue_fringe_merge
from efsm_dpn.learn.guard_inference import synthesize_guard_z3, infer_read_write_sets
from efsm_dpn.models.efsm import EFSM, Variable, Transition, Guard, Update
from collections import defaultdict


def learn_efsm_from_log(
    log_path: str,
    divergence_threshold: float = 0.3,
    use_inductive_miner: bool = False,
) -> EFSM:
    """
    Learn an EFSM from an event log.

    :param log_path: Path to XES or CSV log file.
    :param divergence_threshold: Threshold for state merging compatibility.
    :param use_inductive_miner: Whether to bootstrap from Inductive Miner.
    :return : Learned EFSM.
    :return: Discovered EFSM model.
    """
    df = read_log(log_path)
    traces = extract_traces(df)
    domains = infer_attribute_domains(df)
    propagation = detect_variable_propagation(traces)

    if use_inductive_miner:
        return learn_efsm_from_petri_net(df, traces, domains)
    else:
        return learn_efsm_from_pta(traces, domains, divergence_threshold)


def learn_efsm_from_pta(
    traces: List[List[Tuple[str, Dict[str, Any]]]],
    attribute_domains: Dict[str, Dict[str, Any]],
    divergence_threshold: float = 0.3,
) -> EFSM:
    """
    Learn EFSM via PTA construction and state merging.

    :param traces: List of traces (activity, attributes).
    :param attribute_domains: Domain metadata for attributes.
    :param divergence_threshold: State compatibility threshold.
    :return : Learned EFSM.
    :return: EFSM from PTA approach.
    """
    pta = build_pta(traces)

    attribute_names = list(attribute_domains.keys())
    state_mapping = blue_fringe_merge(pta, attribute_names, divergence_threshold)

    unique_states = set(state_mapping.values())
    states = {f"s{sid}" for sid in unique_states}
    initial = f"s{state_mapping[pta.root.node_id]}"

    variables = {}
    for attr_name, domain in attribute_domains.items():
        dtype = domain.get("dtype", "string")
        if dtype not in ["int", "float", "string", "cat"]:
            dtype = "string"
        variables[attr_name] = Variable(name=attr_name, dtype=dtype)

    transitions: List[Transition] = []
    edge_map: Dict[Tuple[int, str, int], List[Dict[str, Any]]] = defaultdict(list)

    for node in pta.nodes:
        mapped_source = state_mapping.get(node.node_id, node.node_id)
        for label, child in node.children.items():
            mapped_target = state_mapping.get(child.node_id, child.node_id)
            samples = node.edge_samples.get(label, [])
            edge_map[(mapped_source, label, mapped_target)].extend(samples)

    for (source_id, label, target_id), samples in edge_map.items():
        source_state = f"s{source_id}"
        target_state = f"s{target_id}"

        positive_examples = samples
        # Collect negative examples: traces from same source state with different labels
        negative_examples: List[Dict[str, Any]] = []
        for (other_source, other_label, other_target), other_samples in edge_map.items():
            if other_source == source_id and other_label != label:
                negative_examples.extend(other_samples)

        guard = synthesize_guard_z3(
            positive_examples, negative_examples, attribute_domains, max_conjuncts=2
        )

        update_assignments: Dict[str, str] = {}
        for attr_name in attribute_domains.keys():
            if attr_name in [s.get(attr_name) for s in samples if attr_name in s]:
                update_assignments[attr_name] = f"attr.{attr_name}"

        update = Update(assignments=update_assignments)

        transition = Transition(
            source=source_state,
            label=label,
            guard=guard,
            update=update,
            target=target_state,
        )
        transitions.append(transition)

    efsm = EFSM(
        states=states, initial=initial, variables=variables, transitions=transitions
    )

    return efsm


def learn_efsm_from_petri_net(
    df: Any,
    traces: List[List[Tuple[str, Dict[str, Any]]]],
    attribute_domains: Dict[str, Dict[str, Any]],
) -> EFSM:
    """
    Learn EFSM bootstrapped from pm4py Inductive Miner.

    :param df: Event log DataFrame.
    :param traces: List of traces.
    :param attribute_domains: Domain metadata.
    :return : Learned EFSM.
    :return: EFSM bootstrapped from Petri net.
    """
    from efsm_dpn.integration.pm4py_adapter import discover_petri_net_inductive

    petri_net, initial_marking, final_marking = discover_petri_net_inductive(df)

    states = {f"p_{p.name}" for p in petri_net.places}
    initial_place = next(p for p in initial_marking if initial_marking[p] > 0)
    initial = f"p_{initial_place.name}"

    variables = {}
    for attr_name, domain in attribute_domains.items():
        dtype = domain.get("dtype", "string")
        if dtype not in ["int", "float", "string", "cat"]:
            dtype = "string"
        variables[attr_name] = Variable(name=attr_name, dtype=dtype)

    transitions: List[Transition] = []
    for trans in petri_net.transitions:
        in_arcs = [a for a in petri_net.arcs if a.target == trans]
        out_arcs = [a for a in petri_net.arcs if a.source == trans]

        if len(in_arcs) == 1 and len(out_arcs) == 1:
            source_state = f"p_{in_arcs[0].source.name}"
            target_state = f"p_{out_arcs[0].target.name}"
            label = trans.label if trans.label else trans.name

            guard = Guard(expression=None, serialized="true")
            update = Update(assignments={})

            transition = Transition(
                source=source_state,
                label=label,
                guard=guard,
                update=update,
                target=target_state,
            )
            transitions.append(transition)

    efsm = EFSM(
        states=states, initial=initial, variables=variables, transitions=transitions
    )

    return efsm
