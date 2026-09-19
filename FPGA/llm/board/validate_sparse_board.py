#!/usr/bin/env python3
"""Physical sparse-engine validation with uploaded BRAM operands (no DDR claims)."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from pynq import Clocks, MMIO, Overlay

BASE = 0x43C90000
IDENTITY = 0x53505202


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Engine:
    def __init__(self):
        self.mmio = MMIO(BASE, 0x100)
        if self.mmio.read(0x40) != IDENTITY:
            raise RuntimeError('Wrong sparse research ABI: refusing to issue commands')
        self.ctrl = self.mmio.read(0)

    def toggle(self, bit):
        self.ctrl ^= 1 << bit
        self.mmio.write(0, self.ctrl)

    def upload(self, words, weight=False, base=0):
        if self.mmio.read(4) & 1:
            raise RuntimeError('Cannot upload while busy')
        for i, word in enumerate(words):
            self.mmio.write(0x10, base + i)
            self.mmio.write(0x14, int(word))
            self.toggle(2 if weight else 1)

    def run(self, x, weights, shift, skip, reuse):
        workflow_begun = time.monotonic()
        k, rows = len(x), len(weights)
        packed = np.zeros(((k+3)//4)*4, dtype=np.int8)
        packed[:k] = x
        self.upload(packed.view('<u4'))
        upload_seconds = time.monotonic() - workflow_begun
        self.mmio.write(0x08, k)
        self.mmio.write(0x0c, rows | (shift << 8) | (skip << 16) | (reuse << 17))
        begun = time.monotonic()
        self.toggle(0)
        deadline = begun + 5
        while True:
            state = self.mmio.read(4)
            if state & 4:
                raise RuntimeError('Hardware rejected command or concurrent access')
            if state & 2 and not state & 1:
                break
            if time.monotonic() > deadline:
                self.toggle(3)
                raise TimeoutError('Sparse engine timed out and was cancelled')
        elapsed = time.monotonic() - begun
        read_begun = time.monotonic()
        actual = []
        for row in range(rows):
            self.mmio.write(0x18, row)
            # AXI round trip gives the registered result selector time to settle.
            self.mmio.read(0x18)
            value = self.mmio.read(0x54)
            actual.append(value if value < 2**31 else value - 2**32)
        result_read_seconds = time.monotonic() - read_begun
        workflow_seconds = time.monotonic() - workflow_begun
        retained = x.astype(np.int64)
        retained[np.abs(retained) < (1 << shift)] = 0
        expected = weights[:, :k].astype(np.int64).dot(retained)
        expected = ((expected + 2**31) % 2**32 - 2**31).tolist()
        if actual != expected:
            raise AssertionError('Exact INT32 mismatch: actual={} expected={}'.format(actual, expected))
        expected_map = 0
        for block in range((k+63)//64):
            if np.any(retained[block*64:(block+1)*64]):
                expected_map |= 1 << block
        actual_map = self.mmio.read(0x58) | (self.mmio.read(0x5c) << 32)
        if actual_map != expected_map:
            raise AssertionError('Producer map mismatch')
        passes = 1 if reuse else (rows+7)//8
        expected_reads = passes * (bin(expected_map).count('1') if skip else (k+63)//64)
        if self.mmio.read(0x4c) != expected_reads:
            raise AssertionError('Activation read count mismatch')
        return dict(k=k, rows=rows, shift=shift, skip=skip, reuse=reuse,
                    survivors=int(np.count_nonzero(retained)), cycles=self.mmio.read(0x44),
                    preparation_cycles=self.mmio.read(0x48), activation_requests=self.mmio.read(0x4c),
                    weight_requests=self.mmio.read(0x50), scan_cycles=self.mmio.read(0x60),
                    command_wall_seconds=elapsed, activation_upload_seconds=upload_seconds,
                    result_read_seconds=result_read_seconds, workflow_seconds=workflow_seconds,
                    results=actual, expected=expected, passed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bit', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('board_results.json'))
    parser.add_argument('--clock-mhz', type=int, choices=(100, 200), default=200)
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    bit = args.bit.resolve()
    hwh = bit.with_suffix('.hwh')
    if not bit.is_file() or not hwh.is_file():
        raise FileNotFoundError('Matching BIT/HWH required')
    manifest = json.loads((bit.parent / 'release_manifest.json').read_text())
    for path in (bit, hwh):
        if manifest['artifacts'][path.name] != sha(path):
            raise RuntimeError('Artifact hash differs from qualified release')
    if args.clock_mhz > manifest['qualified_clock_mhz']:
        raise RuntimeError('Requested clock exceeds this release timing qualification')
    overlay = Overlay(str(bit), download=True)
    if not overlay.is_loaded():
        raise RuntimeError('Overlay did not load')
    host_clock = None
    if 'clk_fpga_1' in manifest.get('qualified_clocks_mhz', {}):
        Clocks.fclk1_mhz = 100
        host_clock = float(Clocks.fclk1_mhz)
        if abs(host_clock - 100) > 0.1:
            raise RuntimeError('Unexpected host bus clock: {}'.format(host_clock))
    Clocks.fclk0_mhz = args.clock_mhz
    clock = float(Clocks.fclk0_mhz)
    if abs(clock-args.clock_mhz) > 0.1:
        raise RuntimeError('Unexpected actual FCLK0: {}'.format(clock))
    engine = Engine()
    r = np.arange(64, dtype=np.int64)[:, None]
    c = np.arange(4096, dtype=np.int64)[None, :]
    weights = (((r*13+c*7) % 255)-127).astype(np.int8)
    weight_upload_begun = time.monotonic()
    engine.upload(weights.reshape(-1).view('<u4'), weight=True)
    weight_upload_seconds = time.monotonic() - weight_upload_begun
    record = dict(board=platform.node(), clock_mhz=clock, bit_sha256=sha(bit), hwh_sha256=sha(hwh),
                  host_clock_mhz=host_clock,
                  release_host_utc=manifest['host_utc'], qualified_clock_mhz=manifest['qualified_clock_mhz'],
                  weight_upload_seconds=weight_upload_seconds,
                  abi=hex(IDENTITY), memory='CPU-uploaded BRAM; timed region excludes upload', cases=[])
    cases = []
    for density in (1000, 500, 100, 10, 1, 0):
        for mode in range(4):
            cases.append((4096, 64, density, 0, mode & 1, (mode >> 1) & 1))
    cases += [(0,9,0,0,1,1),(3,1,1000,0,1,1),(65,9,500,0,1,1)]
    cases += [(129,17,1000,s,1,1) for s in (1,2,7,8,15)]
    for repeat in range(args.repeat):
        for k, rows, density, shift, skip, reuse in cases:
            idx = np.arange(k, dtype=np.int64)
            x = np.where((idx*137)%1000 < density, (idx*17)%255-128, 0).astype(np.int8)
            result = engine.run(x, weights[:rows], shift, skip, reuse)
            result.update(repeat=repeat, nominal_density_per_mille=density,
                          consumer_us=result['cycles']/clock,
                          preparation_us=result['preparation_cycles']/clock)
            record['cases'].append(result)
            args.output.write_text(json.dumps(record, indent=2))
            print('PASS k={} rows={} density={} skip={} reuse={} cycles={}'.format(
                k, rows, density, skip, reuse, result['cycles']), flush=True)
    print('SPARSE_BOARD_PASS cases={} clock_mhz={}'.format(len(record['cases']), clock), flush=True)


if __name__ == '__main__':
    main()
