"""
Simple example script to discover DPN from the synthetic example log.

Loads the synthetic_small.xes file and performs EFSM/DPN discovery.
"""

from pathlib import Path
from efsm_dpn.learn.efsm_learner import learn_efsm_from_log
from efsm_dpn.map.efsm_to_dpn import map_efsm_to_dpn
from efsm_dpn.integration.pm4py_adapter import export_dpn_to_json


def main():
    # Path to the example XES file
    log_path = Path(__file__).parent / "examples" / "synthetic_small.xes"
    
    print(f"Loading event log: {log_path}")
    
    # Learn EFSM from the event log
    print("Discovering EFSM from event log...")
    efsm = learn_efsm_from_log(
        str(log_path),
        divergence_threshold=0.3,
        use_inductive_miner=False
    )
    
    print(f"✓ Learned EFSM with {len(efsm.states)} states and {len(efsm.transitions)} transitions")
    
    # Map EFSM to DPN
    print("\nMapping EFSM to Data Petri Net...")
    dpn = map_efsm_to_dpn(efsm)
    
    print(f"✓ Created DPN with {len(dpn.petri_net.places)} places and {len(dpn.petri_net.transitions)} transitions")
    
    # Export to JSON
    output_path = Path(__file__).parent / "output" / "discovered_model.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_dpn_to_json(
        dpn, 
        str(output_path),
        name="Discovered DPN from synthetic_small.xes",
        description="Data Petri Net discovered using EFSM learning from event log"
    )
    print(f"\n✓ Exported DPN to: {output_path}")
    
    print("\nDiscovery completed successfully!")
    return efsm, dpn


if __name__ == "__main__":
    efsm, dpn = main()
