#!/bin/bash
# Quick launcher for thesis experiments with nohup

cd "$(dirname "$0")" || exit 1

echo "Starting thesis benchmark experiments..."
echo "Log file: thesis_bench.log"
echo "Start time: $(date)"

# Run all experiments in background
nohup python -u run_all_thesis_experiments.py --pattern "convergence_" > thesis_bench.log 2>&1 &

PID=$!
echo "Process ID: $PID"
echo "To monitor progress: tail -f thesis_bench.log"
echo "To cancel: kill $PID"
