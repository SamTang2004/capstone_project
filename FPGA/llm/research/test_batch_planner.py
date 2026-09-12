import pytest
from batch_planner import Job, plan, execute


def fixture(disjoint=False):
    weights = {'w': tuple(tuple((k+c)%16-8 for c in range(8)) for k in range(32))}
    jobs = [Job(5, r, 'w', tuple(2 if (k//4 == r if disjoint else k<4) else 0
                                 for k in range(32))) for r in range(8)]
    return jobs, weights


@pytest.mark.parametrize('disjoint', [False, True])
@pytest.mark.parametrize('threshold', [0, 4, 12, 16])
def test_golden_and_bounds(disjoint, threshold):
    jobs, weights = fixture(disjoint)
    p = plan(jobs, weights, threshold=threshold)
    got = execute(p, jobs, weights)
    for j, stat in zip(jobs, p['jobs']):
        golden = [sum(a*b[c] for a, b in zip(j.activation, weights['w'])) for c in range(8)]
        assert all(abs(x-y) <= bound for x, y, bound in zip(golden, got[(j.batch,j.task)], stat['error_bound']))
        if threshold == 0:
            assert got[(j.batch,j.task)] == golden


def test_union_not_individual_sparsity_controls_cost():
    aligned = plan(*fixture(False))
    disjoint = plan(*fixture(True))
    assert [j['dropped_k'] for j in aligned['jobs']] == [j['dropped_k'] for j in disjoint['jobs']]
    assert aligned['array_slots'] == 1 and disjoint['array_slots'] == 2
    assert aligned['unfused_dense_slots'] == 16 and aligned['dense_fused_slots'] == 2


def test_budget_and_zero():
    jobs, weights = fixture()
    p = plan(jobs, weights, s=0)
    assert not p['admitted'] and not p['tiles']
    with pytest.raises(ValueError):
        execute(p, jobs, weights)
    assert not plan(jobs, weights, s=1)['admitted']  # packing exceeds remaining budget
    zero = [Job(0, 0, 'w', (0,)*32)]
    p = plan(zero, weights)
    assert p['array_slots'] == 0 and execute(p, zero, weights)[(0,0)] == [0]*8


def test_no_cross_weight_or_batch_fusion():
    jobs, weights = fixture()
    weights['v'] = weights['w']
    jobs[0] = Job(9, 0, 'w', jobs[0].activation)
    jobs[1] = Job(5, 1, 'v', jobs[1].activation)
    p = plan(jobs, weights)
    assert p['array_slots'] == 3


def test_fixed_slots_and_validation():
    jobs, weights = fixture()
    p = plan(jobs, weights)
    assert p['schedule_slots'] * 64 >= p['required_cycles']
    assert all(len(t['a']) == 8 and len(t['b']) == 16 for t in p['tiles'])
    with pytest.raises(ValueError):
        plan(jobs, weights, k0=3)
    with pytest.raises(ValueError):
        plan(jobs+jobs, weights)
