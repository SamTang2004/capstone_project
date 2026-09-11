import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pytest
from budgetcuda.quantum_stealing import (
    QuantumWork, QuantumMachine, evaluate_steals, split_work, slots,
)
from budgetcuda.work_stealing_model import TaskTag


def test_transfer_cost_reverses_steal_decision():
    machine = QuantumMachine(t0=8)
    cheap = QuantumWork('cheap', TaskTag(1, 2, 3), 64, weight_bytes=4096,
                        weight_resident=True)
    expensive = QuantumWork('expensive', TaskTag(), 64, weight_bytes=65536)
    report = evaluate_steals([expensive, cheap], machine, victim_busy=128)
    assert report['selected'] == 'cheap'
    assert not report['candidates'][0]['worth_stealing']
    assert report['candidates'][1]['tag'] == dict(batch_id=1, operation_id=2, generation=3)


def test_busy_link_can_eliminate_benefit():
    work = QuantumWork('w', TaskTag(), 64)
    assert evaluate_steals([work], victim_busy=256)['selected'] == 'w'
    assert evaluate_steals([work], victim_busy=256, link_ready=1000)['selected'] is None


def test_split_preserves_work_and_bytes_but_duplicates_fixed_costs():
    work = QuantumWork('w', TaskTag(2, 3, 1), 101, weight_bytes=1003,
                       activation_bytes=37)
    parts = split_work(work, 32, merge_cycles=2)
    assert [p.compute_cycles for p in parts] == [32, 32, 32, 5]
    assert sum(p.weight_bytes for p in parts) == 1003
    assert sum(p.activation_bytes for p in parts) == 37
    assert sum(p.compute_cycles for p in parts) == 101
    assert all(p.tag == work.tag for p in parts)
    assert sum(p.result_bytes for p in parts) == 4 * work.result_bytes


def test_slots_and_invalid_inputs():
    assert slots(17, 16) == 32
    assert slots(0, 16) == 0
    with pytest.raises(ValueError):
        QuantumMachine(t0=0)
    with pytest.raises(ValueError):
        evaluate_steals([], thief_ready=-1)
