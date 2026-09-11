"""Synthetic transfer/locality sensitivity experiment; not LLM speedup data."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'FPGA'))
from budgetcuda.quantum_stealing import QuantumWork, QuantumMachine, evaluate_steals, split_work
from budgetcuda.work_stealing_model import TaskTag


def main():
    for t0 in (4, 8, 16, 32):
        machine = QuantumMachine(t0=t0)
        queue = [
            QuantumWork('cold_large', TaskTag(0, 0, 0), 128, weight_bytes=65536),
            QuantumWork('resident_medium', TaskTag(1, 0, 0), 64,
                        weight_bytes=4096, weight_resident=True),
            QuantumWork('tiny', TaskTag(2, 0, 0), 4),
        ]
        report = evaluate_steals(queue, machine, victim_busy=128)
        print(json.dumps(dict(t0=t0, **report)))
    parent = QuantumWork('cold', TaskTag(), 256, weight_bytes=16384)
    for chunk in (256, 128, 64, 32, 16):
        parts = split_work(parent, chunk, merge_cycles=2)
        report = evaluate_steals(parts, QuantumMachine(), victim_busy=128)
        print(json.dumps(dict(chunk=chunk, tasks=len(parts), **report)))


if __name__ == '__main__':
    main()
