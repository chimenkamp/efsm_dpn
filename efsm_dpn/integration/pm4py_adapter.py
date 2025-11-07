"""
pm4py integration adapter.

Provides wrappers for pm4py functionality: Inductive Miner, alignments, PNML I/O.

:return : pm4py integration utilities.
:return: Functions for pm4py operations.
"""

from typing import Any, Tuple
import pandas as pd
import pm4py
import json
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from efsm_dpn.models.dpn import DPN
import xml.etree.ElementTree as ET


def discover_petri_net_inductive(
    df: pd.DataFrame,
) -> Tuple[PetriNet, Marking, Marking]:
    """
    Discover Petri net using Inductive Miner.

    :param df: Event log DataFrame.
    :return : Tuple of (petri_net, initial_marking, final_marking).
    :return: Discovered Petri net with markings.
    """
    df_pm4py = dataframe_utils.convert_timestamp_columns_in_df(df)
    df_pm4py = df_pm4py.rename(
        columns={
            "case_id": "case:concept:name",
            "activity": "concept:name",
            "timestamp": "time:timestamp",
        }
    )

    log = log_converter.apply(df_pm4py)
    net, initial_marking, final_marking = inductive_miner.apply(log)

    return net, initial_marking, final_marking


def compute_alignments(
    df: pd.DataFrame, petri_net: PetriNet, initial_marking: Marking, final_marking: Marking
) -> Any:
    """
    Compute alignments for event log against Petri net.

    :param df: Event log DataFrame.
    :param petri_net: Petri net model.
    :param initial_marking: Initial marking.
    :param final_marking: Final marking.
    :return : Alignment results.
    :return: List of alignments per trace.
    """
    df_pm4py = dataframe_utils.convert_timestamp_columns_in_df(df)
    df_pm4py = df_pm4py.rename(
        columns={
            "case_id": "case:concept:name",
            "activity": "concept:name",
            "timestamp": "time:timestamp",
        }
    )

    log = log_converter.apply(df_pm4py)
    aligned_traces = alignments.apply(log, petri_net, initial_marking, final_marking)

    return aligned_traces


def export_dpn_to_pnml(dpn: DPN, filepath: str) -> None:
    """
    Export DPN to PNML file with data annotations.

    :param dpn: Data-Aware Petri Net.
    :param filepath: Output file path.
    :return : None.
    :return: File write side-effect.
    """
    pnml_exporter.apply(dpn.petri_net, dpn.initial_marking, filepath)

    tree = ET.parse(filepath)
    root = tree.getroot()

    ns = {"pnml": "http://www.pnml.org/version-2009/grammar/pnml"}

    for name, dpn_trans in dpn.data_transitions.items():
        for trans_elem in root.findall(".//pnml:transition", ns):
            trans_id = trans_elem.get("id")
            if trans_id == name or trans_elem.find(".//pnml:name/pnml:text", ns).text == name:
                data_elem = ET.SubElement(trans_elem, "data")

                guard_elem = ET.SubElement(data_elem, "guard")
                guard_elem.text = str(dpn_trans.guard)

                update_elem = ET.SubElement(data_elem, "update")
                update_elem.text = str(dpn_trans.update)

                read_elem = ET.SubElement(data_elem, "read")
                read_elem.text = ",".join(sorted(dpn_trans.read_vars))

                write_elem = ET.SubElement(data_elem, "write")
                write_elem.text = ",".join(sorted(dpn_trans.write_vars))

    variables_elem = ET.SubElement(root, "variables")
    for var_name, var_type in dpn.variables.items():
        var_elem = ET.SubElement(variables_elem, "variable")
        var_elem.set("name", var_name)
        var_elem.set("type", var_type)

    tree.write(filepath, encoding="utf-8", xml_declaration=True)


def import_pnml(filepath: str) -> Tuple[PetriNet, Marking, Marking]:
    """
    Import Petri net from PNML file.

    :param filepath: Input file path.
    :return : Tuple of (petri_net, initial_marking, final_marking).
    :return: Loaded Petri net with markings.
    """
    net, initial_marking, final_marking = pnml_importer.apply(filepath)
    return net, initial_marking, final_marking


def export_dpn_to_json(dpn: DPN, filepath: str, name: str = "Discovered DPN", description: str = "Data Petri Net discovered from event log") -> None:
    """
    Export DPN to JSON file with structured format.

    :param dpn: Data-Aware Petri Net.
    :param filepath: Output file path.
    :param name: Name of the Petri net.
    :param description: Description of the Petri net.
    :return : None.
    :return: File write side-effect.
    """
    # Create places
    places = []
    place_index = {}
    for idx, place in enumerate(dpn.petri_net.places):
        place_id = f"P_{place.name}"
        tokens = dpn.initial_marking.get(place, 0) if dpn.initial_marking else 0
        places.append({
            "id": place_id,
            "position": {"x": 0, "y": 0},
            "label": place.name,
            "tokens": tokens,
            "capacity": None,
            "radius": 20
        })
        place_index[place] = place_id
    
    # Create transitions
    transitions = []
    transition_index = {}
    for idx, trans in enumerate(dpn.petri_net.transitions):
        trans_id = f"T_{trans.name}"
        
        # Get guard and update from data transitions
        precondition = ""
        postcondition = ""
        if trans.name in dpn.data_transitions:
            dt = dpn.data_transitions[trans.name]
            precondition = str(dt.guard) if dt.guard else ""
            postcondition = str(dt.update) if dt.update else ""
        
        transitions.append({
            "id": trans_id,
            "position": {"x": 0, "y": 0},
            "label": trans.label if trans.label else trans.name,
            "width": 20,
            "height": 50,
            "isEnabled": False,
            "priority": 1,
            "delay": 0,
            "precondition": precondition,
            "postcondition": postcondition
        })
        transition_index[trans] = trans_id
    
    # Create arcs
    arcs = []
    arc_counter = 1
    for arc in dpn.petri_net.arcs:
        source_id = place_index.get(arc.source) or transition_index.get(arc.source)
        target_id = place_index.get(arc.target) or transition_index.get(arc.target)
        
        if source_id and target_id:
            arcs.append({
                "id": f"A{arc_counter}",
                "source": source_id,
                "target": target_id,
                "weight": arc.weight if hasattr(arc, 'weight') else 1,
                "type": "regular",
                "points": [],
                "label": str(arc.weight) if hasattr(arc, 'weight') else "1"
            })
            arc_counter += 1
    
    # Create data variables
    data_variables = []
    for var_name, var_type in dpn.variables.items():
        # Infer type representation
        json_type = "string"
        if var_type in ["int", "integer"]:
            json_type = "number"
        elif var_type in ["float", "double", "real"]:
            json_type = "number"
        elif var_type in ["bool", "boolean"]:
            json_type = "boolean"
        
        data_variables.append({
            "id": f"var_{var_name}",
            "name": var_name,
            "type": json_type,
            "currentValue": 0 if json_type == "number" else (False if json_type == "boolean" else ""),
            "description": f"Variable {var_name} of type {var_type}"
        })
    
    # Build final structure
    output = {
        "name": name,
        "description": description,
        "places": places,
        "transitions": transitions,
        "arcs": arcs,
        "dataVariables": data_variables
    }
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
