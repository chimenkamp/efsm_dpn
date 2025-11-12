"""
Event log I/O utilities.

Provides functions to read XES/CSV event logs and extract traces with attributes.

:return : Log I/O functions.
:return: Functions for reading logs and extracting traces.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd
import pm4py
from collections import defaultdict


def read_log(filepath: str, log_sample_ratio: float = 0.0) -> pd.DataFrame:
    """
    Read event log from XES or CSV file.

    :param filepath: Path to log file (.xes or .csv).
    :return : DataFrame with case_id, activity, timestamp, and attributes.
    :return: Event log as DataFrame.
    """
    if filepath.endswith(".xes"):
        # pm4py.read_xes() now returns a DataFrame directly
        df = pm4py.read_xes(filepath)
        
        # Standardize column names
        rename_map = {}
        if "concept:name" in df.columns:
            rename_map["concept:name"] = "activity"
        if "case:concept:name" in df.columns:
            rename_map["case:concept:name"] = "case_id"
        if "time:timestamp" in df.columns:
            rename_map["time:timestamp"] = "timestamp"
        
        df = df.rename(columns=rename_map)
        return df
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        

        # Sample the log but keep cases intact
        if 0.0 < log_sample_ratio < 1.0:
            case_ids = df["case_id"].unique()
            sampled_case_ids = pd.Series(case_ids).sample(
                frac=log_sample_ratio
            )
            df = df[df["case_id"].isin(sampled_case_ids)]

        return df
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def extract_traces(
    df: pd.DataFrame,
    case_col: str = "case_id",
    activity_col: str = "activity",
    timestamp_col: str = "timestamp",
) -> List[List[Tuple[str, Dict[str, Any]]]]:
    """
    Extract ordered traces from event log DataFrame.

    :param df: Event log DataFrame.
    :param case_col: Column name for case identifier.
    :param activity_col: Column name for activity.
    :param timestamp_col: Column name for timestamp.
    :return : List of traces, each trace is list of (activity, attributes).
    :return: Traces as ordered event sequences.
    """
    traces = []
    attribute_cols = [
        col
        for col in df.columns
        if col not in [case_col, activity_col, timestamp_col]
    ]

    for case_id, group in df.groupby(case_col):
        if timestamp_col in df.columns:
            group = group.sort_values(timestamp_col)
        trace = []
        for _, row in group.iterrows():
            activity = row[activity_col]
            attrs = {col: row[col] for col in attribute_cols if pd.notna(row[col])}
            trace.append((activity, attrs))
        traces.append(trace)

    return traces


def infer_attribute_domains(
    df: pd.DataFrame,
    case_col: str = "case_id",
    activity_col: str = "activity",
    timestamp_col: str = "timestamp",
) -> Dict[str, Dict[str, Any]]:
    """
    Infer domains and types for event attributes.

    :param df: Event log DataFrame.
    :param case_col: Column name for case identifier.
    :param activity_col: Column name for activity.
    :param timestamp_col: Column name for timestamp.
    :return : Dictionary mapping attribute names to domain info.
    :return: Attribute domain metadata.
    """
    attribute_cols = [
        col
        for col in df.columns
        if col not in [case_col, activity_col, timestamp_col]
    ]
    domains: Dict[str, Dict[str, Any]] = {}

    for col in attribute_cols:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue

        domain_info: Dict[str, Any] = {"name": col}

        if pd.api.types.is_numeric_dtype(df[col]):
            if pd.api.types.is_integer_dtype(df[col]):
                domain_info["dtype"] = "int"
            else:
                domain_info["dtype"] = "float"
            domain_info["min"] = float(non_null.min())
            domain_info["max"] = float(non_null.max())
            domain_info["quantiles"] = non_null.quantile([0.25, 0.5, 0.75]).tolist()
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(
            df[col]
        ):
            unique_vals = non_null.unique()
            if len(unique_vals) <= 20:
                domain_info["dtype"] = "cat"
                domain_info["values"] = list(unique_vals)
            else:
                domain_info["dtype"] = "string"
        else:
            domain_info["dtype"] = "string"

        domains[col] = domain_info

    return domains


def detect_variable_propagation(
    traces: List[List[Tuple[str, Dict[str, Any]]]]
) -> Dict[str, str]:
    """
    Detect which attributes propagate as persistent variables across events.

    :param traces: List of traces (activity, attributes).
    :return : Dictionary mapping variable names to propagation mode.
    :return: Variable propagation patterns.
    """
    attr_persistence: Dict[str, int] = defaultdict(int)
    attr_total: Dict[str, int] = defaultdict(int)

    for trace in traces:
        last_values: Dict[str, Any] = {}
        for activity, attrs in trace:
            for attr, value in attrs.items():
                attr_total[attr] += 1
                if attr in last_values and last_values[attr] == value:
                    attr_persistence[attr] += 1
                last_values[attr] = value

    propagation: Dict[str, str] = {}
    for attr in attr_total:
        if attr_total[attr] > 0:
            persistence_rate = attr_persistence[attr] / attr_total[attr]
            if persistence_rate > 0.7:
                propagation[attr] = "persistent"
            elif persistence_rate > 0.3:
                propagation[attr] = "sometimes"
            else:
                propagation[attr] = "transient"

    return propagation
