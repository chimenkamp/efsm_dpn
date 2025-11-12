"""Analyze the activity flow in the discovered EFSM."""

from pathlib import Path
from efsm_dpn.learn.efsm_learner import learn_efsm_from_log
from collections import defaultdict

log_path = "/Users/christianimenkamp/Documents/Data-Repository/Community/Road-Traffic-Fine-Management-Process/Road_Traffic_Fine_Management_Process.xes"

print(f"Loading and analyzing: {log_path}")

efsm = learn_efsm_from_log(
    str(log_path),
    divergence_threshold=0.7,
    use_inductive_miner=False,
    log_sample_ratio=0.2,
)

print(f"\nEFSM has {len(efsm.states)} states and {len(efsm.transitions)} transitions")

# Build activity graph
activity_graph = defaultdict(set)
for trans in efsm.transitions:
    for next_trans in efsm.transitions:
        if next_trans.source == trans.target:
            activity_graph[trans.label].add(next_trans.label)

# Analyze the flow
print(f"\nActivity flow graph:")
for source, targets in sorted(activity_graph.items()):
    print(f"\n{source} can be followed by:")
    for target in sorted(targets):
        print(f"  -> {target}")
    print(f"  Total successors: {len(targets)}")

# Count total routing transitions needed
total_routing = sum(len(targets) - 1 for targets in activity_graph.values() if len(targets) > 1)
print(f"\nEstimated routing transitions needed: {total_routing}")
print(f"(One routing transition per additional branch in XOR-splits)")
