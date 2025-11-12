"""
State merging for EFSM learning.

Implements blue-fringe style state merging with compatibility checks.

:return : State merging utilities.
:return: Functions for merging compatible PTA states.
"""

from typing import Dict, List, Optional, Set, Tuple
from efsm_dpn.learn.pta import PTA, PTANode
from scipy.spatial.distance import jensenshannon
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO for faster execution
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('state_merger_debug.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def compute_attribute_divergence(
    node1: PTANode, node2: PTANode, attr_name: str
) -> float:
    """
    Compute JS divergence for attribute distributions on outgoing edges.

    :param node1: First PTA node.
    :param node2: Second PTA node.
    :param attr_name: Attribute name.
    :return : Divergence score (0 = identical, 1 = completely different).
    :return: Jensen-Shannon divergence.
    """
    labels1 = set(node1.edge_samples.keys())
    labels2 = set(node2.edge_samples.keys())
    common_labels = labels1 & labels2

    if not common_labels:
        logger.debug(f"      No common labels for attribute '{attr_name}' (node1: {labels1}, node2: {labels2})")
        return 1.0

    divergences = []
    for label in common_labels:
        stats1 = node1.get_edge_statistics(label, attr_name)
        stats2 = node2.get_edge_statistics(label, attr_name)

        if not stats1 or not stats2:
            continue

        if "value_counts" in stats1 and "value_counts" in stats2:
            all_values = set(stats1["value_counts"].keys()) | set(
                stats2["value_counts"].keys()
            )
            total1 = sum(stats1["value_counts"].values())
            total2 = sum(stats2["value_counts"].values())

            if total1 == 0 or total2 == 0:
                continue

            dist1 = np.array(
                [stats1["value_counts"].get(v, 0) / total1 for v in all_values]
            )
            dist2 = np.array(
                [stats2["value_counts"].get(v, 0) / total2 for v in all_values]
            )
            js_div = jensenshannon(dist1, dist2)
            divergences.append(float(js_div))

        elif "mean" in stats1 and "mean" in stats2:
            range_val = max(
                abs(stats1.get("max", 0) - stats1.get("min", 0)),
                abs(stats2.get("max", 0) - stats2.get("min", 0)),
                1.0,
            )
            mean_diff = abs(stats1["mean"] - stats2["mean"]) / range_val
            divergences.append(min(mean_diff, 1.0))

    if not divergences:
        return 0.0

    return float(np.mean(divergences))


def are_states_compatible(
    node1: PTANode,
    node2: PTANode,
    attribute_names: List[str],
    divergence_threshold: float = 0.3,
) -> bool:
    """
    Check if two PTA nodes are compatible for merging.

    :param node1: First PTA node.
    :param node2: Second PTA node.
    :param attribute_names: List of attribute names to check.
    :param divergence_threshold: Maximum allowed divergence.
    :return : True if compatible.
    :return: Compatibility boolean.
    """
    # Check attribute compatibility for common behaviors
    # compute_attribute_divergence already considers only common labels
    for attr_name in attribute_names:
        div = compute_attribute_divergence(node1, node2, attr_name)
        if div > divergence_threshold:
            logger.debug(f"    Incompatible on attribute '{attr_name}': divergence={div:.3f} > {divergence_threshold}")
            return False
        else:
            logger.debug(f"    Compatible on attribute '{attr_name}': divergence={div:.3f}")

    return True


def merge_states(pta: PTA, node1_id: int, node2_id: int) -> Dict[int, int]:
    """
    Merge two nodes and return mapping from old to new node IDs.

    :param pta: Prefix tree acceptor.
    :param node1_id: First node ID (keep this).
    :param node2_id: Second node ID (merge into first).
    :return : Mapping from old to new node IDs.
    :return: Node ID remapping.
    """
    mapping = {node2_id: node1_id}

    node1 = next(n for n in pta.nodes if n.node_id == node1_id)
    node2 = next(n for n in pta.nodes if n.node_id == node2_id)

    for label, child2 in list(node2.children.items()):
        if label in node1.children:
            child1 = node1.children[label]
            sub_mapping = merge_states(pta, child1.node_id, child2.node_id)
            mapping.update(sub_mapping)
        else:
            node1.children[label] = child2

    for label, samples in list(node2.edge_samples.items()):
        node1.edge_samples[label].extend(samples)

    if node2.accepting:
        node1.accepting = True

    return mapping


def blue_fringe_merge(
    pta: PTA, attribute_names: List[str], divergence_threshold: float = 0.3
) -> Dict[int, int]:
    """
    Perform blue-fringe state merging on PTA.

    :param pta: Prefix tree acceptor.
    :param attribute_names: Attributes to consider for compatibility.
    :param divergence_threshold: Compatibility threshold.
    :return : Final mapping from PTA node IDs to merged state IDs.
    :return: Node ID to state ID mapping.
    """
    logger.info(f"Starting blue_fringe_merge with {len(pta.nodes)} PTA nodes")
    logger.info(f"Attributes to consider: {attribute_names}")
    logger.info(f"Divergence threshold: {divergence_threshold}")
    
    red: Set[int] = {pta.root.node_id}
    blue: Set[int] = set()
    mapping: Dict[int, int] = {n.node_id: n.node_id for n in pta.nodes}

    for child in pta.root.children.values():
        blue.add(child.node_id)
    
    logger.info(f"Initial: RED={red}, BLUE size={len(blue)}")

    iteration = 0
    total_merges = 0
    
    while blue:
        iteration += 1
        blue_node_id = next(iter(blue))
        blue_node = next(n for n in pta.nodes if n.node_id == blue_node_id)
        
        logger.debug(f"\n--- Iteration {iteration} ---")
        logger.debug(f"Considering blue node {blue_node_id}")
        logger.debug(f"Blue node has children with labels: {list(blue_node.children.keys())}")

        merged = False
        compatibility_checks = 0
        
        for red_node_id in list(red):
            red_node = next(n for n in pta.nodes if n.node_id == red_node_id)
            compatibility_checks += 1
            
            logger.debug(f"  Checking compatibility with red node {red_node_id}")
            logger.debug(f"  Red node has children with labels: {list(red_node.children.keys())}")
            
            if are_states_compatible(
                red_node, blue_node, attribute_names, divergence_threshold
            ):
                logger.info(f"  ✓ COMPATIBLE! Merging blue {blue_node_id} into red {red_node_id}")
                
                # Identify which children of blue_node were NEW (not matched with red_node children)
                new_children_from_merge = set()
                for label, child in blue_node.children.items():
                    if label not in red_node.children:
                        # This child from blue will be added to red as a new branch
                        new_children_from_merge.add(child.node_id)
                
                merge_mapping = merge_states(pta, red_node_id, blue_node_id)
                logger.debug(f"  Merge mapping: {merge_mapping}")
                
                for old_id, new_id in merge_mapping.items():
                    mapping[old_id] = new_id
                blue.discard(blue_node_id)
                
                # Add the NEW children (that weren't recursively merged) to BLUE
                for child_id in new_children_from_merge:
                    if child_id not in red and child_id not in blue:
                        blue.add(child_id)
                        logger.debug(f"  Added new branch child {child_id} to BLUE")
                
                merged = True
                total_merges += 1
                break
            else:
                logger.debug(f"  ✗ Not compatible with red {red_node_id}")

        if not merged:
            logger.debug(f"Blue node {blue_node_id} not compatible with any red node (checked {compatibility_checks} nodes)")
            logger.debug(f"Promoting blue {blue_node_id} to RED")
            blue.discard(blue_node_id)
            red.add(blue_node_id)
            for child in blue_node.children.values():
                if child.node_id not in red and child.node_id not in blue:
                    blue.add(child.node_id)
                    logger.debug(f"  Added child {child.node_id} to BLUE")
        
        if iteration % 100 == 0:
            logger.info(f"Progress: iteration {iteration}, RED size={len(red)}, BLUE size={len(blue)}, total merges={total_merges}")

    logger.info(f"\nMerging complete after {iteration} iterations")
    logger.info(f"Total merges performed: {total_merges}")
    logger.info(f"Final RED size: {len(red)}")
    logger.info(f"Nodes before mapping cleanup: {len(set(mapping.values()))}")
    
    # Apply transitive closure: ensure all mappings point to final representative
    for node_id in list(mapping.keys()):
        representative = mapping[node_id]
        while representative in mapping and mapping[representative] != representative:
            representative = mapping[representative]
        mapping[node_id] = representative

    unique_states = len(set(mapping.values()))
    logger.info(f"Unique states after transitive closure: {unique_states}")
    logger.info(f"Reduction: {len(pta.nodes)} -> {unique_states} ({100*(1-unique_states/len(pta.nodes)):.1f}% reduction)")

    return mapping
