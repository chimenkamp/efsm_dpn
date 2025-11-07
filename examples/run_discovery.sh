#!/bin/bash

# Example script for running EFSM-DPN discovery on synthetic data

echo "=== EFSM-DPN Discovery Example ==="
echo ""

# Create output directory
mkdir -p ../out

# Discover EFSM/DPN from log
echo "Step 1: Discovering EFSM/DPN from log..."
efsm-dpn discover \
    --log synthetic_small.xes \
    --out-pnml ../out/model_dpn.pnml \
    --out-efsm ../out/model_efsm.json \
    --divergence-threshold 0.3

echo ""
echo "Step 2: Evaluating conformance..."
efsm-dpn evaluate \
    --log synthetic_small.xes \
    --pnml ../out/model_dpn.pnml \
    --out-json ../out/conformance_results.json

echo ""
echo "Step 3: Simulating EFSM on traces..."
efsm-dpn simulate \
    --efsm ../out/model_efsm.json \
    --log synthetic_small.xes \
    --max-traces 5 \
    --verbose

echo ""
echo "=== Done ==="
echo "Output files in ../out/"
