"""
Prefix Tree Acceptor (PTA) construction.

Builds a prefix tree from traces with statistical annotations on edges.

:return : PTA construction utilities.
:return: Functions and classes for PTA building.
"""

from typing import Any, Dict, List, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class PTANode:
    """
    Node in a Prefix Tree Acceptor.

    :param node_id: Unique identifier for node.
    :param depth: Depth in tree.
    :param accepting: Whether this is an accepting state (end of trace).
    :param children: Map from activity labels to child nodes.
    :param edge_samples: Samples of attributes observed on outgoing edges.
    :return : PTANode instance.
    :return: A PTA node.
    """

    node_id: int
    depth: int
    accepting: bool = False
    children: Dict[str, "PTANode"] = field(default_factory=dict)
    edge_samples: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))

    def add_edge_sample(self, label: str, attrs: Dict[str, Any]) -> None:
        """
        Record attribute sample for an outgoing edge.

        :param label: Activity label.
        :param attrs: Event attributes.
        :return : None.
        :return: Side-effect of recording sample.
        """
        self.edge_samples[label].append(attrs)

    def get_edge_statistics(self, label: str, attr_name: str) -> Dict[str, Any]:
        """
        Compute statistics for an attribute on an edge.

        :param label: Activity label.
        :param attr_name: Attribute name.
        :return : Statistics dictionary.
        :return: Mean, std, quantiles, or value counts.
        """
        samples = self.edge_samples.get(label, [])
        values = [s.get(attr_name) for s in samples if attr_name in s]
        if not values:
            return {}

        stats: Dict[str, Any] = {"count": len(values)}

        # Check if numeric (excluding booleans, since bool is subclass of int)
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            numeric_vals = np.array(values)
            stats["mean"] = float(np.mean(numeric_vals))
            stats["std"] = float(np.std(numeric_vals))
            stats["min"] = float(np.min(numeric_vals))
            stats["max"] = float(np.max(numeric_vals))
            stats["quantiles"] = np.quantile(numeric_vals, [0.25, 0.5, 0.75]).tolist()
        else:
            value_counts = defaultdict(int)
            for v in values:
                value_counts[str(v)] += 1
            stats["value_counts"] = dict(value_counts)

        return stats


class PTA:
    """
    Prefix Tree Acceptor.

    :param root: Root node of the tree.
    :param nodes: List of all nodes.
    :param node_counter: Counter for generating node IDs.
    :return : PTA instance.
    :return: A prefix tree acceptor.
    """

    def __init__(self) -> None:
        """
        Initialize empty PTA.

        :return : None.
        :return: Initialized PTA.
        """
        self.root = PTANode(node_id=0, depth=0)
        self.nodes: List[PTANode] = [self.root]
        self.node_counter = 1

    def add_trace(self, trace: List[Tuple[str, Dict[str, Any]]]) -> None:
        """
        Add a trace to the PTA.

        :param trace: List of (activity, attributes) tuples.
        :return : None.
        :return: Side-effect of adding trace.
        """
        current = self.root
        for activity, attrs in trace:
            current.add_edge_sample(activity, attrs)
            if activity not in current.children:
                new_node = PTANode(
                    node_id=self.node_counter, depth=current.depth + 1
                )
                self.node_counter += 1
                current.children[activity] = new_node
                self.nodes.append(new_node)
            current = current.children[activity]
        current.accepting = True

    def get_reachable_states(self, node: PTANode) -> Set[int]:
        """
        Get all states reachable from a node.

        :param node: Starting node.
        :return : Set of node IDs.
        :return: Reachable state set.
        """
        reachable = {node.node_id}
        for child in node.children.values():
            reachable.update(self.get_reachable_states(child))
        return reachable

    def get_future_labels(self, node: PTANode) -> Set[str]:
        """
        Get set of labels reachable from a node.

        :param node: Starting node.
        :return : Set of activity labels.
        :return: Future labels from node.
        """
        labels: Set[str] = set(node.children.keys())
        for child in node.children.values():
            labels.update(self.get_future_labels(child))
        return labels


def build_pta(traces: List[List[Tuple[str, Dict[str, Any]]]]) -> PTA:
    """
    Build a Prefix Tree Acceptor from traces.

    :param traces: List of traces (activity, attributes).
    :return : PTA instance.
    :return: Constructed prefix tree acceptor.
    """
    pta = PTA()
    for trace in traces:
        pta.add_trace(trace)
    return pta
