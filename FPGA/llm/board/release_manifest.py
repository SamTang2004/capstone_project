#!/usr/bin/env python3
"""Record the exact board artifacts and uncommitted HDL used for a release."""
import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument('release', type=Path)
args = parser.parse_args()
release = args.release.resolve()
timing = (release / 'timing_impl.rpt').read_text()
clock_match = re.search(r'^clk_fpga_0\s+\{[^}]+\}\s+[\d.]+\s+([\d.]+)', timing, re.M)
if not clock_match or 'All user specified timing constraints are met.' not in timing:
    raise RuntimeError('A passing routed timing report is required')
if re.search(r'^clk_fpga_1\s+\{', timing, re.M):
    skew = (release / 'bus_skew.rpt').read_text()
    if 'Slack (MET)' not in skew or 'Slack (VIOLATED)' in skew:
        raise RuntimeError('Passing clock-crossing bus skew checks are required')
    cdc = (release / 'cdc.rpt').read_text()
    if re.search(r'^CDC-\d+\s+(Warning|Critical)', cdc, re.M):
        raise RuntimeError('CDC warnings require review before release')
sources = list((root / 'FPGA/llm/rtl').glob('*.sv'))
sources += [root / p for p in (
    'FPGA/llm/rtl/llm_sparse_board_ip.v',
    'FPGA/hardware/rtl/axi_lite_control.sv',
    'FPGA/hardware/vivado/pynq_z2_timing_margin.xdc',
    'FPGA/llm/vivado/build_sparse_board.tcl',
    'FPGA/llm/vivado/sparse_board_timing.xdc',
    'FPGA/llm/board/validate_sparse_board.py',
)]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
record = dict(
    host_utc=datetime.now(timezone.utc).isoformat(),
    qualified_clock_mhz=float(clock_match.group(1)),
    qualified_clocks_mhz={name: float(freq) for name, freq in re.findall(
        r'^(clk_fpga_\d+)\s+\{[^}]+\}\s+[\d.]+\s+([\d.]+)', timing, re.M)},
    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).splitlines(),
    sources={p.relative_to(root).as_posix(): digest(p) for p in sorted(sources)},
    artifacts={p.name: digest(p) for p in sorted(release.iterdir())
               if p.is_file() and p.suffix in ('.bit', '.hwh', '.xsa', '.rpt')},
    scope='CPU-uploaded BRAM operands; upload excluded from hardware counters',
)
(release / 'release_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
print(release / 'release_manifest.json')
