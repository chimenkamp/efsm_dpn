"""
Command-line interface for EFSM-DPN.

Provides commands for discovery, evaluation, and simulation.

:return : CLI commands.
:return: Main entry point for command-line usage.
"""

import argparse
import json
import sys
from pathlib import Path
from efsm_dpn.learn.efsm_learner import learn_efsm_from_log
from efsm_dpn.map.efsm_to_dpn import map_efsm_to_dpn
from efsm_dpn.integration.pm4py_adapter import export_dpn_to_pnml, import_pnml
from efsm_dpn.conformance.checks import evaluate_conformance
from efsm_dpn.logs.io import read_log


def cmd_discover(args: argparse.Namespace) -> None:
    """
    Discover EFSM/DPN from event log.

    :param args: Command-line arguments.
    :return : None.
    :return: Side-effect of discovery and export.
    """
    print(f"Learning EFSM from log: {args.log}")
    print(f"Bootstrap from Inductive Miner: {args.bootstrap_inductive_miner}")

    efsm = learn_efsm_from_log(
        args.log,
        divergence_threshold=args.divergence_threshold,
        use_inductive_miner=args.bootstrap_inductive_miner,
    )

    print(f"Learned EFSM with {len(efsm.states)} states and {len(efsm.transitions)} transitions")

    dpn = map_efsm_to_dpn(efsm)
    print(f"Mapped to DPN with {len(dpn.petri_net.places)} places and {len(dpn.petri_net.transitions)} transitions")

    out_path = Path(args.out_pnml)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_dpn_to_pnml(dpn, args.out_pnml)
    print(f"Exported DPN to: {args.out_pnml}")

    if args.out_efsm:
        efsm.to_json(args.out_efsm)
        print(f"Exported EFSM to: {args.out_efsm}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """
    Evaluate conformance of log against DPN.

    :param args: Command-line arguments.
    :return : None.
    :return: Side-effect of conformance evaluation.
    """
    print(f"Evaluating conformance: log={args.log}, pnml={args.pnml}")

    from efsm_dpn.models.dpn import DPN
    from efsm_dpn.integration.pm4py_adapter import import_pnml

    net, initial_marking, final_marking = import_pnml(args.pnml)

    dpn = DPN(
        petri_net=net,
        initial_marking=initial_marking,
        final_marking=final_marking,
    )

    results = evaluate_conformance(dpn, args.log)

    print("\n=== Conformance Results ===")
    print(f"Number of traces: {results['num_traces']}")
    print(f"Control-flow fitness: {results['control_flow_fitness'].get('fitness', 0.0):.3f}")
    print(f"Guard satisfaction rate: {results['guard_satisfaction']['satisfaction_rate']:.3f}")
    print(f"  Satisfied: {results['guard_satisfaction']['satisfied']}")
    print(f"  Violated: {results['guard_satisfaction']['violated']}")
    print(f"  Undefined: {results['guard_satisfaction']['undefined']}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to: {args.out_json}")


def cmd_simulate(args: argparse.Namespace) -> None:
    """
    Simulate EFSM on event log traces.

    :param args: Command-line arguments.
    :return : None.
    :return: Side-effect of simulation.
    """
    print(f"Simulating EFSM from: {args.efsm}")

    from efsm_dpn.models.efsm import EFSM
    from efsm_dpn.logs.io import read_log, extract_traces

    efsm = EFSM.from_json(args.efsm)
    df = read_log(args.log)
    traces = extract_traces(df)

    accepted = 0
    rejected = 0

    for i, trace in enumerate(traces[:args.max_traces]):
        result, path, final_state = efsm.simulate_trace(trace)
        if result:
            accepted += 1
        else:
            rejected += 1

        if args.verbose:
            print(f"Trace {i}: {'ACCEPTED' if result else 'REJECTED'}")
            print(f"  Path: {' -> '.join(path)}")

    print(f"\n=== Simulation Results ===")
    print(f"Accepted: {accepted}/{len(traces[:args.max_traces])}")
    print(f"Rejected: {rejected}/{len(traces[:args.max_traces])}")


def main() -> None:
    """
    Main entry point for CLI.

    :return : None.
    :return: Exit code.
    """
    parser = argparse.ArgumentParser(
        description="EFSM-DPN: Data-Aware Process Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    discover_parser = subparsers.add_parser("discover", help="Discover EFSM/DPN from log")
    discover_parser.add_argument("--log", required=True, help="Path to event log (XES/CSV)")
    discover_parser.add_argument("--out-pnml", required=True, help="Output PNML file path")
    discover_parser.add_argument("--out-efsm", help="Output EFSM JSON file path")
    discover_parser.add_argument(
        "--bootstrap-inductive-miner",
        action="store_true",
        help="Bootstrap from pm4py Inductive Miner",
    )
    discover_parser.add_argument(
        "--divergence-threshold",
        type=float,
        default=0.3,
        help="State merging divergence threshold (default: 0.3)",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate conformance")
    evaluate_parser.add_argument("--log", required=True, help="Path to event log")
    evaluate_parser.add_argument("--pnml", required=True, help="Path to PNML model")
    evaluate_parser.add_argument("--out-json", help="Output JSON file for results")

    simulate_parser = subparsers.add_parser("simulate", help="Simulate EFSM on log")
    simulate_parser.add_argument("--efsm", required=True, help="Path to EFSM JSON file")
    simulate_parser.add_argument("--log", required=True, help="Path to event log")
    simulate_parser.add_argument(
        "--max-traces", type=int, default=10, help="Maximum traces to simulate"
    )
    simulate_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "discover":
            cmd_discover(args)
        elif args.command == "evaluate":
            cmd_evaluate(args)
        elif args.command == "simulate":
            cmd_simulate(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
