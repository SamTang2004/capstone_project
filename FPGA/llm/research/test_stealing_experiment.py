import pytest
from stealing_experiment import Work, simulate, workloads, quantize


def test_quantize():
    assert [quantize(x, 16) for x in (0, 1, 16, 17)] == [0, 16, 16, 32]


@pytest.mark.parametrize('name', workloads())
@pytest.mark.parametrize('policy', ['fifo', 'naive', 'aware'])
def test_conservation(name, policy):
    tasks = workloads()[name]
    r = simulate(tasks, policy, t0=16)
    assert r['compute_cycles'] == sum(t.compute for t in tasks)
    assert sorted((t['uid'], t['batch']) for t in r['trace']) == sorted((t.uid, t.batch) for t in tasks)
    for w in (0, 1):
        events = sorted((t['start'], t['finish']) for t in r['trace'] if t['worker'] == w)
        assert all(a[1] <= b[0] for a, b in zip(events, events[1:]))
    assert 0 <= r['useful_lane_util'] <= r['compute_util'] <= 1


def test_resident_steal_and_cold_rejection():
    cases = workloads()
    assert simulate(cases['skew_resident'], 'aware')['makespan'] < simulate(cases['skew_resident'])['makespan']
    cold = simulate(cases['skew_cold'], 'aware')
    assert cold['steals'] == 0 and cold['rejected_probes'] > 0 and cold['probe_cycles'] > 0
    assert simulate(cases['skew_cold'], 'naive')['makespan'] > cold['makespan']


def test_estimation_does_not_pad_arithmetic():
    tasks = [Work(0, 5, 0, 16, 0)]
    assert simulate(tasks, t0=64)['makespan'] == 17
    assert simulate(tasks, t0=64, padding=True)['makespan'] == 64
    assert simulate(tasks, t0=64, padding=True)['padding_cycles'] == 47


def test_gemv_lanes_are_not_filled_by_time_balancing():
    r = simulate(workloads()['gemv_skew'], 'aware')
    assert r['useful_lane_util'] == r['compute_util'] / 8


def test_invalid_and_empty():
    assert simulate([])['makespan'] == 0
    with pytest.raises(ValueError):
        simulate([Work(0, 0, 0, 1, 0)] * 2)
