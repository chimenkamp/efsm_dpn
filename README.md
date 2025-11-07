# EFSM-DPN: Extended Finite State Machine Discovery and Data Petri Net Mapping

## Overview

Automated discovery of Extended Finite State Machines (EFSMs) from event logs and their subsequent mapping to Data-Aware Petri Nets (DPNs). The approach combines state-based automata learning with constraint synthesis to produce process models that capture both control-flow and data-flow behavior.

## Formal Definitions

### Extended Finite State Machine

An EFSM is defined as a tuple `M = ⟨S, s₀, X, Σ, T⟩` where:

- `S`: finite set of control states
- `s₀ ∈ S`: initial state
- `X`: finite set of typed variables (int, float, string, categorical)
- `Σ`: alphabet of event labels (activity names)
- `T ⊆ S × Σ × G(X) × U(X) × S`: set of transitions

Each transition `t ∈ T` consists of:
- Source state `s_src ∈ S`
- Event label `a ∈ Σ`
- Guard predicate `g ∈ G(X)`: boolean expression over variables
- Update function `u ∈ U(X)`: variable assignments
- Target state `s_tgt ∈ S`

### Data-Aware Petri Net

A DPN extends classical Petri nets with data semantics. It is a tuple `N = ⟨P, T, F, m₀, X, G, U⟩` where:

- `P`: finite set of places
- `T`: finite set of transitions
- `F ⊆ (P × T) ∪ (T × P)`: flow relation
- `m₀: P → ℕ`: initial marking
- `X`: finite set of typed variables
- `G: T → Predicates(X)`: guard function mapping transitions to predicates
- `U: T → Assignments(X)`: update function mapping transitions to variable assignments

## Theoretical Foundation

The approach draws from grammatical inference and automata learning, specifically:

1. **Prefix Tree Acceptor (PTA) construction**: Builds initial hypothesis automaton accepting all observed traces
2. **State merging with compatibility testing**: Generalizes the automaton using evidence-based state equivalence
3. **Guard synthesis via constraint solving**: Learns predicates that discriminate between alternative transitions using SAT-based search
4. **EFSM-to-DPN structural mapping**: Translates state-based models to place/transition nets preserving data semantics

This differs from pure control-flow mining (Inductive Miner, Alpha algorithm) by explicitly modeling data dependencies and from pure trace clustering by maintaining a formal automaton structure.

## Pipeline Architecture

### Stage 1: Log Preprocessing

**Input**: Event log (XES or CSV format)

**Operations**:
- Parse events into case-grouped traces
- Infer attribute domains and types (numeric vs categorical)
- Detect variable propagation patterns across events

**Output**: Structured trace representation `[(activity, attributes)]` with domain metadata

**Implementation**: `efsm_dpn.logs.io`

### Stage 2: PTA Construction

**Algorithm**:
```
PTA(traces):
  root ← new node(id=0, depth=0)
  for each trace in traces:
    current ← root
    for each (activity, attrs) in trace:
      record attrs as edge sample on current →activity
      if activity ∉ current.children:
        current.children[activity] ← new node
      current ← current.children[activity]
    mark current as accepting
```

**Properties**:
- Overgeneralizes: accepts exactly the observed traces
- Node count: O(Σ |trace|) in worst case
- Each edge annotated with attribute value samples

**Implementation**: `efsm_dpn.learn.pta.build_pta`

### Stage 3: State Merging

**Algorithm**: Blue-Fringe variant

```
BlueFringe(pta, attributes, threshold):
  red ← {root}
  blue ← children(root)
  mapping ← identity
  
  while blue ≠ ∅:
    select b ∈ blue
    merged ← false
    
    for each r ∈ red:
      if Compatible(r, b, attributes, threshold):
        Merge(r, b)
        update mapping
        merged ← true
        break
    
    if not merged:
      red ← red ∪ {b}
      blue ← blue ∪ children(b) \ red
    blue ← blue \ {b}
```

**Compatibility Test**:

Two nodes `n₁`, `n₂` are compatible if:

1. Outgoing edge labels match: `Labels(n₁) = Labels(n₂)`
2. For each attribute `x` and label `a`, the distributions of `x` on edge `n₁ →ᵃ` and `n₂ →ᵃ` satisfy:
   - Categorical: Jensen-Shannon divergence `JS(P₁‖P₂) < threshold`
   - Numeric: normalized mean difference `|μ₁ - μ₂| / range < threshold`

**Complexity**: O(|nodes|² × |attributes| × |labels|)

**Implementation**: `efsm_dpn.learn.state_merger.blue_fringe_merge`

### Stage 4: Guard Synthesis

**Problem Formulation**:

Given:
- Positive examples `E⁺`: attribute valuations that enable transition `t`
- Negative examples `E⁻`: attribute valuations from same source state that enable different transitions

Find: Predicate `φ(X)` such that:
- `∀e ∈ E⁺: φ(e) = true`
- `∀e ∈ E⁻: φ(e) = false`

**Algorithm**:

```
SynthesizeGuard(E⁺, E⁻, domains, max_conjuncts):
  predicates ← GenerateAtomicPredicates(E⁺, E⁻, domains)
  
  for k from 1 to max_conjuncts:
    for subset p₁,...,pₖ in predicates:
      candidate ← p₁ ∧ p₂ ∧ ... ∧ pₖ
      if Validate(candidate, E⁺, E⁻):
        return candidate
  
  return true  // unconstrained guard
```

**Atomic Predicate Generation**:

For numeric attributes:
- Threshold predicates: `x ≤ θ`, `x ≥ θ`, `x < θ`, `x > θ`, `x = θ`
- Thresholds derived from: quantile boundaries between `E⁺` and `E⁻`, domain percentiles, decision boundary values

For categorical attributes:
- Equality predicates: `x = v` for each `v` in domain

**Validation** uses Z3 SMT solver to check satisfiability.

**Implementation**: `efsm_dpn.learn.guard_inference.synthesize_guard_z3`

### Stage 5: EFSM-to-DPN Mapping

**Mapping Rules**:

1. **States to Places**: `∀s ∈ S: create place p_s`
2. **Transitions to Transitions**: `∀t ∈ T: create transition t_pn`
3. **Flow Arcs**: 
   - `∀t = (s₁, a, g, u, s₂): add arc p_{s₁} → t → p_{s₂}`
4. **Data Annotations**:
   - Attach guard `g(t)` to transition
   - Attach update `u(t)` to transition
   - Compute read/write sets from guard and update expressions

**Preservation Properties**:
- Reachability equivalence: state `s` reachable in EFSM ⟺ place `p_s` reachable in DPN
- Trace equivalence: execution sequence valid in EFSM ⟺ firing sequence valid in DPN (under same data valuation)

**Implementation**: `efsm_dpn.map.efsm_to_dpn.map_efsm_to_dpn`

## Usage

### Discovery from Event Log

```python
from efsm_dpn.learn.efsm_learner import learn_efsm_from_log
from efsm_dpn.map.efsm_to_dpn import map_efsm_to_dpn
from efsm_dpn.integration.pm4py_adapter import export_dpn_to_json

# Learn EFSM
efsm = learn_efsm_from_log(
    "event_log.xes",
    divergence_threshold=0.3,      # state compatibility threshold
    use_inductive_miner=False      # use PTA-based approach
)

# Map to DPN
dpn = map_efsm_to_dpn(efsm)

# Export
export_dpn_to_json(dpn, "model.json")
```

### Conformance Checking

```python
from efsm_dpn.conformance.checks import evaluate_conformance

metrics = evaluate_conformance(dpn, "event_log.xes")
# Returns: control-flow fitness, guard satisfaction rate
```

## Parameters

### `divergence_threshold` (default: 0.3)

Controls state merging aggressiveness. Lower values produce larger, more precise models. Higher values produce smaller, more generalized models.

- Range: [0.0, 1.0]
- 0.0: merge only identical distributions (maximum precision)
- 1.0: merge all states (maximum generalization)

### `max_conjuncts` (default: 2)

Limits complexity of synthesized guard predicates. Higher values enable more expressive guards but increase search space exponentially.

- Range: [1, ∞)
- Recommended: 2-3 for interpretability

### `use_inductive_miner` (default: False)

- `False`: Use PTA construction and state merging (recommended for data-aware discovery)
- `True`: Bootstrap control-flow from Inductive Miner, then add data predicates

## Model Serialization

### JSON Format

```json
{
  "petri_net": {
    "places": [...],
    "transitions": [...]
  },
  "data_transitions": {
    "t0_approve": {
      "guard": "amount <= 1000",
      "update": {"status": "approved"},
      "read_vars": ["amount"],
      "write_vars": ["status"]
    }
  },
  "variables": {
    "amount": "int",
    "status": "cat"
  }
}
```

### PNML Export

Data annotations embedded in PNML using custom `<data>` elements on transitions. Compatible with standard Petri net tools for control-flow analysis.

## Dependencies

Core:
- `pm4py`: Petri net infrastructure, process mining algorithms
- `z3-solver`: SMT solving for guard synthesis
- `pandas`, `numpy`: Data manipulation
- `scipy`: Statistical computations (Jensen-Shannon divergence)

## Limitations

1. **Guard synthesis**: Limited to conjunctions of atomic predicates. Does not synthesize disjunctions or complex nested formulas.

2. **State merging**: Greedy algorithm without backtracking. Merge decisions are irreversible and order-dependent.

3. **Update inference**: Currently uses simple attribute propagation. Does not learn arithmetic expressions or complex transformations.

4. **Scalability**: PTA construction is linear in trace count, but state merging is quadratic in states. Guard synthesis is exponential in predicate count.

5. **Noise handling**: No explicit outlier detection. Noisy attribute values may prevent state merging or produce overly specific guards.

## Future Extensions

- Disjunctive guard synthesis using CEGIS (counterexample-guided inductive synthesis)
- Probabilistic compatibility metrics for noisy logs
- Incremental learning for streaming event data
- Integration with declare constraints for hybrid models

## References

Theoretical basis combines techniques from:
- Grammatical inference (state merging, blue-fringe algorithm)
- Program synthesis (constraint-based predicate learning)
- Process mining (Petri net discovery, conformance checking)
- Data-aware process modeling (guard/update semantics)

## License

See LICENSE file.
