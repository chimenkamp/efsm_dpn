"""
Data-aware conformance checking.

Evaluates control-flow fitness and data guard satisfaction.

:return : Conformance checking utilities.
:return: Functions for evaluating DPN conformance.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd
from pm4py.objects.petri_net.obj import PetriNet, Marking
from efsm_dpn.models.dpn import DPN
from efsm_dpn.logs.io import read_log, extract_traces
from efsm_dpn.integration.pm4py_adapter import compute_alignments


def evaluate_conformance(dpn: DPN, log_path: str) -> Dict[str, Any]:
    """
    Evaluate conformance of event log against DPN.

    :param dpn: Data-Aware Petri Net.
    :param log_path: Path to event log file.
    :return : Conformance metrics.
    :return: Dictionary with fitness and guard satisfaction metrics.
    """
    df = read_log(log_path)
    traces = extract_traces(df)

    control_fitness = evaluate_control_flow_fitness(dpn, df)

    guard_metrics = evaluate_guard_satisfaction(dpn, df, traces)

    results = {
        "control_flow_fitness": control_fitness,
        "guard_satisfaction": guard_metrics,
        "num_traces": len(traces),
    }

    return results


def evaluate_control_flow_fitness(dpn: DPN, df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate control-flow fitness using alignments.

    :param dpn: Data-Aware Petri Net.
    :param df: Event log DataFrame.
    :return : Control-flow fitness metrics.
    :return: Fitness score and statistics.
    """
    try:
        aligned_traces = compute_alignments(
            df, dpn.petri_net, dpn.initial_marking, dpn.final_marking or Marking()
        )

        if not aligned_traces:
            return {"fitness": 0.0, "num_aligned": 0}

        fitness_scores = []
        for alignment in aligned_traces:
            if "fitness" in alignment:
                fitness_scores.append(alignment["fitness"])

        avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0

        return {
            "fitness": avg_fitness,
            "num_aligned": len(aligned_traces),
        }
    except Exception as e:
        return {"fitness": 0.0, "num_aligned": 0, "error": str(e)}


def evaluate_guard_satisfaction(
    dpn: DPN, df: pd.DataFrame, traces: List[List[Tuple[str, Dict[str, Any]]]]
) -> Dict[str, Any]:
    """
    Evaluate data-aware guard satisfaction on traces.

    :param dpn: Data-Aware Petri Net.
    :param df: Event log DataFrame.
    :param traces: List of traces.
    :return : Guard satisfaction metrics.
    :return: Satisfaction rates and violation details.
    """
    total_transitions = 0
    satisfied_guards = 0
    violated_guards = 0
    undefined_guards = 0

    violation_details: Dict[str, int] = {}

    for trace in traces:
        var_state: Dict[str, Any] = {var: None for var in dpn.variables.keys()}

        for activity, attrs in trace:
            candidates = [
                (name, dt)
                for name, dt in dpn.data_transitions.items()
                if dt.pn_transition.label == activity
            ]

            for trans_name, dpn_trans in candidates:
                total_transitions += 1
                try:
                    if dpn_trans.guard.evaluate(var_state):
                        satisfied_guards += 1
                        var_state = dpn_trans.update.apply(var_state, attrs)
                    else:
                        violated_guards += 1
                        violation_details[trans_name] = (
                            violation_details.get(trans_name, 0) + 1
                        )
                except Exception:
                    undefined_guards += 1

    satisfaction_rate = (
        satisfied_guards / total_transitions if total_transitions > 0 else 0.0
    )

    return {
        "satisfaction_rate": satisfaction_rate,
        "total_transitions": total_transitions,
        "satisfied": satisfied_guards,
        "violated": violated_guards,
        "undefined": undefined_guards,
        "violation_details": violation_details,
    }
