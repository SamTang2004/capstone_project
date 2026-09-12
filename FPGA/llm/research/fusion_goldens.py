"""Compile deterministic W4A8 tasks, emit XSim vectors, verify tagged outputs."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from batch_planner import Job, plan, execute


def fixtures():
    rng = random.Random(20260912)
    cases = []
    for name, count, k, n, kind in [
        ('aligned_sparse', 24, 32, 8, 'aligned'),
        ('disjoint_sparse', 16, 32, 8, 'disjoint'),
        ('mixed_sparse_dense', 25, 48, 8, 'mixed'),
        ('dense_control', 17, 32, 8, 'dense'),
        ('zero_elision', 9, 32, 8, 'zero'),
        ('odd_k_tail_columns', 19, 33, 3, 'mixed'),
        ('weight_and_batch_isolation', 24, 32, 8, 'isolated'),
        ('signed_extrema', 9, 18, 8, 'extrema'),
    ]:
        weights = {'w': tuple(tuple(rng.randrange(-8, 8) for _ in range(n)) for _ in range(k))}
        if kind == 'isolated':
            weights['other'] = tuple(tuple(rng.randrange(-8, 8) for _ in range(n)) for _ in range(k))
        if kind == 'extrema':
            weights['w'] = tuple(tuple(-8 if (i+c)%2 else 7 for c in range(n)) for i in range(k))
        jobs = []
        for row in range(count):
            a = [rng.choice((-7, -3, 1, 4, 9)) for _ in range(k)]
            if kind == 'zero': a = [0]*k
            if kind == 'aligned': a = [x if i<4 else 0 for i,x in enumerate(a)]
            if kind == 'disjoint': a = [x if i//4 == row%8 else 0 for i,x in enumerate(a)]
            if kind == 'mixed' and row%3: a = [x if rng.random()<0.2 else 0 for x in a]
            if kind == 'extrema': a = [-128 if (i+row)%2 else 127 for i in range(k)]
            batch = row//8 if kind == 'isolated' else 0
            task = row%8 if kind == 'isolated' else row
            wid = 'other' if kind == 'isolated' and row%2 else 'w'
            jobs.append(Job(batch, task, wid, tuple(a)))
        cases.append((name, jobs, weights))
    return cases


def dense(j, weights):
    b = weights[j.weight_id]
    return [sum(j.activation[k]*b[k][c] for k in range(len(b))) for c in range(len(b[0]))]


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def generate(out):
    out.mkdir(parents=True, exist_ok=True)
    programs, vectors, software_trace = [], [], []
    tile_id = 0
    for name, jobs, weights in fixtures():
        p = plan(jobs, weights, s=4096, threshold=0)
        assert p['admitted']
        golden = [dense(j, weights) for j in jobs]  # independent, original K order
        software = execute(p, jobs, weights)
        assert [software[(j.batch,j.task)] for j in jobs] == golden
        program = dict(name=name, jobs=[asdict(j) for j in jobs], weights=weights,
                       dense_golden=golden, tiles=[], classification=p['jobs'],
                       sparse_slots=p['array_slots'], dense_fused_slots=p['dense_fused_slots'],
                       unfused_slots=p['unfused_dense_slots'], output=golden)
        for tile in p['tiles']:
            rows = tile['rows']
            cols = len(weights[tile['weight_id']][0])
            partial = [sum(tile['a'][r][k]*tile['b'][k][c] for k in range(16))
                       for r in range(8) for c in range(8)]
            assert all(-(1<<31) <= v < (1<<31) for v in partial)
            program['tiles'].append(dict(tile_id=tile_id, rows=rows,
                tags=[[jobs[i].batch, jobs[i].task] for i in rows],
                weight_id=tile['weight_id'], k_indices=tile['k_indices'], partial=partial))
            vectors.append(f'{tile_id} {jobs[rows[0]].batch} {len(rows)} {cols} 16')
            for k in range(16):
                a = sum((tile['a'][r][k]&255) << (r*8) for r in range(8))
                b = sum((tile['b'][k][c]&15) << (c*4) for c in range(8))
                vectors.append(f'{a:016x} {b:016x}')
            vectors.extend(str(v) for v in partial)
            software_trace.append(dict(event='tile_execute', case=name, tile_id=tile_id,
                tags=program['tiles'][-1]['tags'], k_indices=tile['k_indices'], output=partial))
            tile_id += 1
        for j, output in zip(jobs, golden):
            software_trace.append(dict(event='job_complete', case=name, batch=j.batch, task=j.task, output=output))
        programs.append(program)
    dump(out/'program.json', dict(schema=1, precision='W4A8_INT32', threshold=0,
         k0=16, timing='software trace has ordering only, not cycle measurements', programs=programs))
    (out/'vectors.txt').write_text(str(tile_id)+'\n'+'\n'.join(vectors)+'\n', encoding='ascii')
    (out/'software_trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in software_trace), encoding='utf-8')
    print(f'GENERATED cases={len(programs)} jobs={sum(len(p["jobs"]) for p in programs)} tiles={tile_id}')


def verify(out):
    programs = json.loads((out/'program.json').read_text())['programs']
    trace = [json.loads(line) for line in (out/'rtl_trace.jsonl').read_text().splitlines()]
    lookup = {r['tile_id']: r for r in trace}
    ids = [t['tile_id'] for p in programs for t in p['tiles']]
    assert len(lookup) == len(trace) and set(lookup) == set(ids), 'missing/duplicate/extra tile'
    reports = []
    for p in programs:
        actual = [[0]*len(g) for g in p['dense_golden']]
        for tile in p['tiles']:
            r = lookup[tile['tile_id']]
            assert r['batch'] == tile['tags'][0][0]
            assert r['output'] == tile['partial'], 'RTL fragment mismatch'
            for lane, idx in enumerate(tile['rows']):
                for c in range(len(actual[idx])):
                    actual[idx][c] += r['output'][lane*8+c]
        assert actual == p['dense_golden'], f'final dense golden mismatch: {p["name"]}'
        reports.append(dict(case=p['name'], jobs=len(actual), checked_outputs=sum(map(len,actual)),
            max_abs_error=0, sparse_tiles=p['sparse_slots'], dense_fused_tiles=p['dense_fused_slots'],
            unfused_tiles=p['unfused_slots'], actual=actual))
    summary = dict(result='PASS', source='XSim real DSP48E1 tile, software scatter/merge', cases=reports,
        tile_count=len(trace), execution_cycles=sorted({r['execution_cycles'] for r in trace}),
        input_wait_cycles=sorted({r['input_wait_cycles'] for r in trace}),
        files={f:hashlib.sha256((out/f).read_bytes()).hexdigest()
               for f in ('program.json','vectors.txt','software_trace.jsonl','rtl_trace.jsonl')})
    dump(out/'verified_results.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('files','cases')}))
    for r in reports: print({k:v for k,v in r.items() if k!='actual'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['generate','verify'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (generate if args.action == 'generate' else verify)(args.output)
