"""Quick test to verify DPN structure."""

from pathlib import Path
from efsm_dpn.learn.efsm_learner import learn_efsm_from_log
from efsm_dpn.map.efsm_to_dpn import map_efsm_to_dpn
from efsm_dpn.integration.pm4py_adapter import export_dpn_to_json

# Use the small example file
log_path = Path(__file__).parent / "examples" / "synthetic_small.xes"

print(f"Loading event log: {log_path}")

# Learn EFSM from the event log
print("Discovering EFSM...")
efsm = learn_efsm_from_log(
    str(log_path),
    divergence_threshold=0.7,
    use_inductive_miner=False,
    log_sample_ratio=1.0,  # Use all data from small file
)

print(f"✓ Learned EFSM with {len(efsm.states)} states and {len(efsm.transitions)} transitions")

# Map EFSM to DPN
print("\nMapping EFSM to Data Petri Net...")
dpn = map_efsm_to_dpn(efsm)

# Count visible vs silent transitions
visible = [t for t in dpn.petri_net.transitions if t.label is not None]
silent = [t for t in dpn.petri_net.transitions if t.label is None]
unique_labels = set(t.label for t in visible)

print(f"\nDPN Structure:")
print(f"  Total places: {len(dpn.petri_net.places)}")
print(f"  Total transitions: {len(dpn.petri_net.transitions)}")
print(f"  Visible transitions: {len(visible)}")
print(f"  Silent transitions: {len(silent)}")
print(f"  Unique activity labels: {len(unique_labels)}")
print(f"  Labels: {sorted(unique_labels)}")

# Export to JSON
output_path = Path(__file__).parent / "output" / "test_model.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

export_dpn_to_json(
    dpn, 
    str(output_path),
    name="Test DPN",
    description="Test model"
)

print(f"\n✓ Exported to: {output_path}")
