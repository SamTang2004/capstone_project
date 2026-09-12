import json
import pytest
from fusion_goldens import generate, verify


def software_trace_fixture(out):
    generate(out)
    p = json.loads((out/'program.json').read_text())
    records = [dict(tile_id=t['tile_id'], batch=t['tags'][0][0], output=t['partial'],
                    execution_cycles=20, input_wait_cycles=0)
               for case in p['programs'] for t in case['tiles']]
    return records


def write_trace(out, records):
    # Parser unit fixture only, not claimed to be an RTL execution.
    (out/'rtl_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))


def test_reproducible_compilation_and_verification(tmp_path):
    a,b = tmp_path/'a',tmp_path/'b'
    records = software_trace_fixture(a)
    generate(b)
    assert (a/'program.json').read_bytes() == (b/'program.json').read_bytes()
    assert (a/'vectors.txt').read_bytes() == (b/'vectors.txt').read_bytes()
    write_trace(a, records)
    verify(a)
    result = json.loads((a/'verified_results.json').read_text())
    assert len(result['cases']) == 8 and result['tile_count'] == 50
    assert sum(c['jobs'] for c in result['cases']) == 143


@pytest.mark.parametrize('fault', ['missing','duplicate','partial','dense','batch'])
def test_fail_closed_on_corruption(tmp_path, fault):
    records = software_trace_fixture(tmp_path)
    if fault == 'missing': records.pop()
    if fault == 'duplicate': records.append(records[0])
    if fault == 'partial': records[0]['output'][0] += 1
    if fault == 'batch': records[0]['batch'] += 1
    if fault == 'dense':
        p = json.loads((tmp_path/'program.json').read_text())
        p['programs'][0]['dense_golden'][0][0] += 1
        (tmp_path/'program.json').write_text(json.dumps(p))
    write_trace(tmp_path, records)
    with pytest.raises(AssertionError):
        verify(tmp_path)
