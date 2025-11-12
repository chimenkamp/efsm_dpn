"""
Guard inference using Z3 constraint synthesis.

Learns predicates from positive and negative examples of edge usage.

:return : Guard inference utilities.
:return: Functions for synthesizing guards with Z3.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import z3
import numpy as np
import logging
from efsm_dpn.models.efsm import Guard
from collections import defaultdict

logger = logging.getLogger(__name__)


def extract_edge_examples(
    traces: List[List[Tuple[str, Dict[str, Any]]]],
    state_mapping: Dict[int, int],
    source_state: int,
    target_activity: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract positive and negative examples for an edge.

    :param traces: List of traces with state annotations.
    :param state_mapping: Mapping from trace positions to states.
    :param source_state: Source state ID.
    :param target_activity: Activity label of the edge.
    :return : Tuple of (positive_examples, negative_examples).
    :return: Training examples for guard learning.
    """
    positive_examples: List[Dict[str, Any]] = []
    negative_examples: List[Dict[str, Any]] = []

    return positive_examples, negative_examples


def generate_atomic_predicates(
    attr_name: str,
    attr_type: str,
    domain_info: Dict[str, Any],
    positive_examples: List[Dict[str, Any]],
    negative_examples: List[Dict[str, Any]],
) -> List[z3.ExprRef]:
    """
    Generate candidate atomic predicates for an attribute.

    :param attr_name: Attribute name.
    :param attr_type: Attribute type (int, float, cat, string).
    :param domain_info: Domain metadata from log analysis.
    :param positive_examples: Positive training examples.
    :param negative_examples: Negative training examples.
    :return : List of Z3 atomic predicates.
    :return: Candidate predicates.
    """
    predicates: List[z3.ExprRef] = []
    z3_var = None

    if attr_type in ["int", "float"]:
        z3_var = z3.Int(attr_name) if attr_type == "int" else z3.Real(attr_name)

        pos_values = [
            ex.get(attr_name) for ex in positive_examples if attr_name in ex
        ]
        neg_values = [
            ex.get(attr_name) for ex in negative_examples if attr_name in ex
        ]

        all_values = pos_values + neg_values
        if all_values:
            thresholds = []
            
            # Add boundary values between positive and negative examples
            if pos_values and neg_values:
                pos_min, pos_max = min(pos_values), max(pos_values)
                neg_min, neg_max = min(neg_values), max(neg_values)
                
                # Add potential decision boundaries
                if pos_max < neg_min:
                    # Positive values are all less than negative values
                    thresholds.append((pos_max + neg_min) / 2)
                elif neg_max < pos_min:
                    # Negative values are all less than positive values
                    thresholds.append((neg_max + pos_min) / 2)
                
                # Add min and max of each group
                thresholds.extend([pos_min, pos_max, neg_min, neg_max])
            
            # Add domain quantiles if available
            if "quantiles" in domain_info:
                thresholds.extend(domain_info["quantiles"])
            
            # Add percentiles from actual examples
            if pos_values:
                thresholds.extend(np.percentile(pos_values, [25, 50, 75]).tolist())
            if neg_values:
                thresholds.extend(np.percentile(neg_values, [25, 50, 75]).tolist())
            
            # Add all unique values if there aren't too many
            unique_vals = sorted(set(all_values))
            if len(unique_vals) <= 10:
                thresholds.extend(unique_vals)

            # Limit to top 20 most relevant thresholds to avoid explosion
            unique_thresholds = sorted(set(thresholds))
            if len(unique_thresholds) > 20:
                # Sample evenly across the range
                indices = np.linspace(0, len(unique_thresholds) - 1, 20, dtype=int)
                unique_thresholds = [unique_thresholds[i] for i in indices]
            
            for threshold in unique_thresholds:
                if attr_type == "int":
                    # Only add <= and >= to reduce predicate count
                    predicates.append(z3_var <= int(threshold))
                    predicates.append(z3_var >= int(threshold))
                else:
                    predicates.append(z3_var <= threshold)
                    predicates.append(z3_var >= threshold)

    elif attr_type == "cat":
        if "values" in domain_info:
            z3_var = z3.String(attr_name)
            # Limit categorical predicates to top 10 most common values
            values = domain_info["values"]
            if len(values) > 10:
                # Only use first 10 values (assumes they're ordered by frequency)
                values = values[:10]
            for value in values:
                predicates.append(z3_var == z3.StringVal(str(value)))

    return predicates


def synthesize_guard_z3(
    positive_examples: List[Dict[str, Any]],
    negative_examples: List[Dict[str, Any]],
    attribute_domains: Dict[str, Dict[str, Any]],
    max_conjuncts: int = 3,
) -> Optional[Guard]:
    """
    Synthesize guard predicate using Z3 SAT-based search.

    :param positive_examples: Examples that should satisfy guard.
    :param negative_examples: Examples that should violate guard.
    :param attribute_domains: Domain metadata for attributes.
    :param max_conjuncts: Maximum number of atomic predicates in conjunction.
    :return : Synthesized Guard or None.
    :return: Learned guard predicate.
    """
    if not positive_examples:
        logger.debug("No positive examples, returning 'true' guard")
        return Guard(expression=None, serialized="true")

    if not negative_examples:
        logger.debug("No negative examples, returning 'true' guard")
        return Guard(expression=None, serialized="true")

    all_attrs = set()
    for ex in positive_examples + negative_examples:
        all_attrs.update(ex.keys())

    # Separate numerical and categorical predicates to prefer numerical ones
    numerical_predicates: List[z3.ExprRef] = []
    categorical_predicates: List[z3.ExprRef] = []
    
    for attr in sorted(all_attrs):  # Sort for deterministic behavior
        if attr not in attribute_domains:
            continue
        domain = attribute_domains[attr]
        preds = generate_atomic_predicates(
            attr, domain["dtype"], domain, positive_examples, negative_examples
        )
        # Prefer numerical predicates over categorical
        if domain["dtype"] in ["int", "float"]:
            numerical_predicates.extend(preds)
        else:
            categorical_predicates.extend(preds)
    
    # Try numerical predicates first, then categorical
    all_predicates = numerical_predicates + categorical_predicates

    if not all_predicates:
        logger.debug("No predicates generated, returning 'true' guard")
        return Guard(expression=None, serialized="true")

    logger.debug(f"Generated {len(all_predicates)} predicates ({len(numerical_predicates)} numerical, {len(categorical_predicates)} categorical)")
    
    for num_conjuncts in range(1, min(max_conjuncts + 1, len(all_predicates) + 1)):
        for i in range(len(all_predicates)):
            if num_conjuncts == 1:
                candidate = all_predicates[i]
            else:
                conjuncts = all_predicates[i : i + num_conjuncts]
                if len(conjuncts) < num_conjuncts:
                    continue
                candidate = z3.And(*conjuncts)

            if validate_guard(candidate, positive_examples, negative_examples):
                logger.debug(f"Found valid guard with {num_conjuncts} conjunct(s): {str(candidate)[:100]}")
                return Guard(expression=candidate, serialized=str(candidate))

    logger.debug("No valid guard found, returning 'true' guard")
    return Guard(expression=None, serialized="true")


def validate_guard(
    guard_expr: z3.ExprRef,
    positive_examples: List[Dict[str, Any]],
    negative_examples: List[Dict[str, Any]],
) -> bool:
    """
    Validate guard against positive and negative examples.

    :param guard_expr: Z3 guard expression.
    :param positive_examples: Should satisfy guard.
    :param negative_examples: Should not satisfy guard.
    :return : True if guard correctly classifies examples.
    :return: Validation result.
    """
    # Sample examples if there are too many to speed up validation
    pos_sample = positive_examples if len(positive_examples) <= 50 else positive_examples[:50]
    neg_sample = negative_examples if len(negative_examples) <= 50 else negative_examples[:50]
    
    solver = z3.Solver()
    # Set timeout to avoid hanging on complex constraints
    solver.set("timeout", 5000)  # 5 second timeout

    for ex in pos_sample:
        solver.push()
        substituted = guard_expr
        for attr, value in ex.items():
            if isinstance(value, int):
                z3_var = z3.Int(attr)
                substituted = z3.substitute(substituted, (z3_var, z3.IntVal(value)))
            elif isinstance(value, float):
                z3_var = z3.Real(attr)
                substituted = z3.substitute(substituted, (z3_var, z3.RealVal(value)))
            elif isinstance(value, str):
                z3_var = z3.String(attr)
                substituted = z3.substitute(substituted, (z3_var, z3.StringVal(value)))

        solver.add(substituted)
        result = solver.check()
        if result != z3.sat:
            solver.pop()
            return False
        solver.pop()

    for ex in neg_sample:
        solver.push()
        substituted = guard_expr
        for attr, value in ex.items():
            if isinstance(value, int):
                z3_var = z3.Int(attr)
                substituted = z3.substitute(substituted, (z3_var, z3.IntVal(value)))
            elif isinstance(value, float):
                z3_var = z3.Real(attr)
                substituted = z3.substitute(substituted, (z3_var, z3.RealVal(value)))
            elif isinstance(value, str):
                z3_var = z3.String(attr)
                substituted = z3.substitute(substituted, (z3_var, z3.StringVal(value)))

        solver.add(substituted)
        result = solver.check()
        if result == z3.sat:
            solver.pop()
            return False
        solver.pop()

    return True


def infer_read_write_sets(
    guard: Guard, update_assignments: Dict[str, str]
) -> Tuple[Set[str], Set[str]]:
    """
    Infer read and write variable sets from guard and update.

    :param guard: Guard predicate.
    :param update_assignments: Variable update assignments.
    :return : Tuple of (read_vars, write_vars).
    :return: Variable read/write sets.
    """
    read_vars: Set[str] = set()
    write_vars: Set[str] = set(update_assignments.keys())

    if guard.serialized and guard.serialized != "true":
        guard_str = guard.serialized
        for word in guard_str.split():
            if word.isidentifier() and not word in ["And", "Or", "Not"]:
                read_vars.add(word)

    for expr in update_assignments.values():
        for word in expr.split():
            if word.isidentifier() and word not in ["attr"]:
                read_vars.add(word)

    return read_vars, write_vars
