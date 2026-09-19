"""Deterministic architectural model; NOT RTL simulation or measured FPGA timing.

Run: python FPGA/llm/research/activation_sharing_experiment.py
Tests: python -m unittest discover -s FPGA/llm/research -p test_activation_sharing.py
"""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import csv
import random

ROOT = Path(__file__).resolve().parent


def ceildiv(a, b):
    return (a + b - 1) // b


def int32(value):
    return (value + 2**31) % 2**32 - 2**31


@dataclass(frozen=True)
class Config:
    activation_bw: int = 8
    banks: int = 8
    window: int = 64
    response_capacity: int = 64
    memory_latency: int = 2
    layout: str = "row_major"

    def __post_init__(self):
        assert self.activation_bw in (8, 64)
        assert self.banks > 0 and self.window > 0
        assert self.response_capacity >= 8 and self.memory_latency >= 1
        assert self.layout in ("row_major", "output_interleaved")


def find_activations(x, shift, cfg):
    """64-position frames, six-cycle classification delay, eight-entry drain.

    Eight frame slots include assembly, pipeline and serialized output. Input
    assembly overlaps output drain. These are model parameters, not a faithful
    reproduction of the existing RTL pipeline's ready/valid implementation.
    """
    assert 0 <= shift <= 15
    frames = deque()
    assembled = 0
    cursor = 0
    cycle = 0
    indices = []
    while cursor < len(x) or assembled or frames:
        if frames and frames[0][0] <= cycle:
            ready_at, kept = frames[0]
            indices.extend(kept[:8])
            if len(kept) <= 8:
                frames.popleft()
            else:
                frames[0] = (ready_at, kept[8:])
        if cursor < len(x) and len(frames) < 8:
            n = min(cfg.activation_bw, 64 - assembled, len(x) - cursor)
            assembled += n
            cursor += n
            if assembled == 64 or cursor == len(x):
                kept = [k for k in range(cursor - assembled, cursor)
                        if abs(x[k]) >> shift]
                frames.append((cycle + 6, kept))
                assembled = 0
        cycle += 1
    return indices, cycle


def address(row, k, rows, length, layout):
    return row * length + k if layout == "row_major" else k * rows + row


def execute(x, weights, indices, cfg, order="replay", sparse=True):
    """Eight contexts per wave, finite request/response queues and bank service.

    One byte read per bank per cycle, up to eight requests and MACs per cycle.
    A greedy scheduler inspects only the finite pending window. Descriptor
    admission is at most eight products/cycle. Broadcast uses one descriptor
    per activation per wave; replay uses one per product. Bank responses have
    fixed latency. Sparse zero weights are discarded AFTER their charged read.
    """
    rows, length = len(weights), len(x)
    result = [0] * rows
    cycles = reads = issued = useful = descriptor_reads = bank_limited = 0
    for first in range(0, rows, 8):
        stop = min(rows, first + 8)
        if order == "replay":
            tokens = [(r, k) for r in range(first, stop) for k in indices]
            descriptor_reads += len(tokens)
        else:
            assert order == "broadcast"
            tokens = [(r, k) for k in indices for r in range(first, stop)]
            descriptor_reads += len(indices)
        pending = []
        flight = deque()
        ready = deque()
        cursor = 0
        t = 0
        seen = set()
        while cursor < len(tokens) or pending or flight or ready:
            while flight and flight[0][0] <= t:
                _, r, k = flight.popleft()
                if not sparse or weights[r][k] != 0:
                    ready.append((r, k))
            for _ in range(min(8, len(ready))):
                r, k = ready.popleft()
                assert (r, k) not in seen
                seen.add((r, k))
                product = x[k] * weights[r][k]
                result[r] = int32(result[r] + product)
                issued += 1
                useful += product != 0
            used = set()
            retained = []
            slots = cfg.response_capacity - len(flight) - len(ready)
            candidates = len(pending)
            for r, k in pending:
                bank = address(r, k, rows, length, cfg.layout) % cfg.banks
                if bank not in used and len(used) < min(8, slots):
                    used.add(bank)
                    flight.append((t + cfg.memory_latency, r, k))
                    reads += 1
                else:
                    retained.append((r, k))
            bank_limited += candidates >= 8 and slots >= 8 and len(used) < 8
            pending = retained
            n = min(8, cfg.window - len(pending), len(tokens) - cursor)
            pending.extend(tokens[cursor:cursor + n])
            cursor += n
            assert len(pending) <= cfg.window
            assert len(flight) + len(ready) <= cfg.response_capacity
            t += 1
        # Command, empty-wave marker (when needed), and arithmetic drain.
        cycles += t + 4 + (not tokens)
    assert reads == rows * len(indices)
    return dict(compute_cycles=cycles, weight_bytes=reads, issued_macs=issued,
                useful_macs=useful, descriptor_reads=descriptor_reads,
                bank_limited_cycles=bank_limited, result=result)


def experiment(x, weights, shift, cfg):
    rows = len(weights)
    assert rows > 0 and all(len(w) == len(x) for w in weights)
    assert all(-128 <= v <= 127 for v in x)
    assert all(-128 <= v <= 127 for w in weights for v in w)
    kept, find_cycles = find_activations(x, shift, cfg)
    filtered = [v if abs(v) >> shift else 0 for v in x]
    golden = [int32(sum(a * b for a, b in zip(filtered, w))) for w in weights]
    replay = execute(filtered, weights, kept, cfg)
    broadcast = execute(filtered, weights, kept, cfg, "broadcast")
    dense_options = [execute(filtered, weights, list(range(len(x))), cfg, o, False)
                     for o in ("replay", "broadcast")]
    dense = min(dense_options, key=lambda v: v["compute_cycles"])
    index_bytes = max(1, ceildiv(max(1, (len(x) - 1).bit_length()), 8))
    outputs = []
    for name, backend, passes in [("dense_cached", dense, 0),
                                  ("repeated_finder", replay, rows),
                                  ("shared_replay", replay, 1),
                                  ("shared_broadcast", broadcast, 1)]:
        assert backend["result"] == golden, (name, backend["result"], golden)
        frontend = passes * find_cycles if passes else ceildiv(len(x), cfg.activation_bw)
        data = {k: v for k, v in backend.items() if k != "result"}
        data.update(mode=name, rows=rows, k=len(x), kept=len(kept), shift=shift,
                    activation_bw=cfg.activation_bw, layout=cfg.layout,
                    banks=cfg.banks, window=cfg.window,
                    finder_passes=passes, frontend_cycles=frontend,
                    cycles=frontend + backend["compute_cycles"],
                    # Optimistic bound if phases could overlap perfectly;
                    # excludes precedence and finite-buffer penalties.
                    perfect_overlap_bound=max(frontend, backend["compute_cycles"]),
                    activation_source_bytes=len(x) * max(1, passes),
                    descriptor_write_bytes=passes * len(kept) * (index_bytes + 1),
                    descriptor_read_bytes=(backend["descriptor_reads"] * (index_bytes + 1)
                                           if passes else 0),
                    descriptor_store_bytes=(min(8, rows) if passes > 1 else 1)
                                           * len(kept) * (index_bytes + 1) if passes else 0)
        data["source_bytes"] = data["activation_source_bytes"] + data["weight_bytes"]
        data["useful_utilization"] = backend["useful_macs"] / (8 * data["cycles"])
        outputs.append(data)
    return outputs


def fixture(k, rows, density, seed=17, pattern="random", weight_density=1.0, w4=False):
    rng = random.Random(seed)
    indices = list(range(k))
    if pattern == "random":
        rng.shuffle(indices)
    elif pattern == "bank_clustered":
        indices.sort(key=lambda i: (i % 8, i))
    else:
        assert pattern == "clustered"
    keep = set(indices[:round(k * density)])
    x = [rng.choice((-128, -63, -1, 1, 7, 127)) if k in keep else 0 for k in range(k)]
    lo, hi = (-8, 7) if w4 else (-128, 127)
    choices = [v for v in range(lo, hi + 1) if v]
    weights = [[rng.choice(choices) if rng.random() < weight_density else 0
                for _ in range(k)] for _ in range(rows)]
    return x, weights


def report(records):
    path = ROOT / "ACTIVATION_SHARING_PERFORMANCE.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    def select(density, rows, bw=8, layout="row_major", pattern="random"):
        return {r["mode"]: r for r in records if r["density"] == density
                and r["rows"] == rows and r["activation_bw"] == bw
                and r["layout"] == layout and r["pattern"] == pattern and r["seed"] == 17}
    lines = ["# Activation sharing: architectural model results", "",
             "**Exploratory Python model, not RTL or board measurements.** All modeled outputs "
             "match independent scalar INT32 matrix-vector results. See [model contract](ACTIVATION_SHARING.md).", "",
             "## Same memory layout and product order: sharing alone", "",
             "K=4096, eight output rows, eight activation bytes/cycle, eight single-read weight banks, "
             "row-major W8 weights with no zeros, random fixed-count activation masks, seed 17. "
             "Phases are sequential; all activation preparation is charged.", "",
             "| Activation density | Dense cached cycles | Repeated finder | Shared replay | Sharing speedup | Shared useful utilization |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for density in (1.0, .5, .1, .01, .001, 0.0):
        d = select(density, 8)
        a, b = d["repeated_finder"], d["shared_replay"]
        lines.append(f'| {density:.1%} | {d["dense_cached"]["cycles"]} | {a["cycles"]} | {b["cycles"]} | {a["cycles"]/b["cycles"]:.2f}x | {b["useful_utilization"]:.2%} |')
    lines += ["", "## Reuse scaling at 10% activation density", "",
              "Same row-major configuration. Rows above eight execute as sequential eight-context waves; "
              "the shared descriptors persist across waves.", "",
              "| Output rows | Repeated cycles | Shared cycles | Speedup | Activation source bytes, repeated / shared |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for rows in (1, 8, 32, 64):
        d = select(.1, rows)
        a, b = d["repeated_finder"], d["shared_replay"]
        lines.append(f'| {rows} | {a["cycles"]} | {b["cycles"]} | {a["cycles"]/b["cycles"]:.2f}x | {a["activation_source_bytes"]} / {b["activation_source_bytes"]} |')
    lines += ["", "## Layout and broadcast at eight output rows / 10% density", "",
              "Both weight layouts are pre-existing model inputs. Transposing or reloading a matrix is NOT free "
              "and is not modeled. Compare replay/broadcast within each layout before comparing layouts.", "",
              "| Layout | Shared replay cycles | Shared broadcast cycles | Descriptor read bytes, replay / broadcast | Weight bytes (both) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for layout in ("row_major", "output_interleaved"):
        d = select(.1, 8, layout=layout)
        a, b = d["shared_replay"], d["shared_broadcast"]
        lines.append(f'| {layout} | {a["cycles"]} | {b["cycles"]} | {a["descriptor_read_bytes"]} / {b["descriptor_read_bytes"]} | {a["weight_bytes"]} |')
    lines += ["", "## Sensitivity at eight rows / 10% density", "",
              "Same row-major layout; ranges cover seeds 17, 29 and 43. Bank-clustered masks "
              "concentrate surviving indices in one weight bank. Wider activation supply reduces "
              "the preparation cost available to amortize.", "",
              "| Activation pattern | Activation bytes/cycle | Shared replay speedup over repeated finding |",
              "| --- | ---: | ---: |"]
    for pattern, bw in (("random", 8), ("random", 64), ("clustered", 8), ("bank_clustered", 8)):
        ratios = []
        for seed in (17, 29, 43):
            d = {r["mode"]: r for r in records if r["rows"] == 8 and r["density"] == .1
                 and r["layout"] == "row_major" and r["activation_bw"] == bw
                 and r["pattern"] == pattern and r["seed"] == seed}
            ratios.append(d["repeated_finder"]["cycles"] / d["shared_replay"]["cycles"])
        lines.append(f'| {pattern} | {bw} | {min(ratios):.2f}–{max(ratios):.2f}x |')
    d = select(.1, 8)
    a, b = d["repeated_finder"], d["shared_replay"]
    lines += ["", "## Bounds and limitations", "",
              f'- If repeated preparation and compute overlapped perfectly, its optimistic bound is '
              f'{a["perfect_overlap_bound"]} cycles, versus {b["cycles"]} for sequential shared replay '
              f'({a["perfect_overlap_bound"]/b["cycles"]:.2f}x). This is a sensitivity bound, not an implemented baseline.',
              f'- The 10% / eight-row shared descriptor store is {b["descriptor_store_bytes"]} logical bytes '
              '(index plus activation); physical banking, replicated ports and BRAM rounding are uncosted.',
              '- Repeated finding and shared replay issue exactly the same weight requests and useful MACs. '
              'Sharing alone saves activation reads and classification, not weight bytes.',
              '- Source bytes mean local operand-memory reads, not DDR transactions. Descriptor traffic is reported '
              'separately. Weight loading, bursts, DMA, row-result writes and external bandwidth are outside this model.',
              '- Dense control caches the activation once and uses the faster of replay/broadcast schedules for each layout. '
              'A shared dense bypass therefore remains attractive at high density.',
              '- No LUT, power, synthesis, or 100 MHz qualification is claimed for the proposed sharing frontend. '
              'The previous finder timing result does not transfer to this architecture.',
              f'- Full CSV contains {len(records)} configurations, including three seeds, clustered masks, '
              'bank-clustered masks, 8/64-byte activation supply, two layouts, and 1/8/32/64 output rows.', ""]
    (ROOT / "ACTIVATION_SHARING_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    records = []
    for seed in (17, 29, 43):
        for density in (1.0, .5, .1, .01, .001, 0.0):
            for rows in (1, 8, 32, 64):
                x, w = fixture(4096, rows, density, seed)
                for bw in (8, 64):
                    for layout in ("row_major", "output_interleaved"):
                        for r in experiment(x, w, 0, Config(activation_bw=bw, layout=layout)):
                            records.append(dict(seed=seed, density=density, pattern="random", **r))
        for pattern in ("clustered", "bank_clustered"):
            x, w = fixture(4096, 8, .1, seed, pattern)
            for layout in ("row_major", "output_interleaved"):
                for r in experiment(x, w, 0, Config(layout=layout)):
                    records.append(dict(seed=seed, density=.1, pattern=pattern, **r))
        print(f"Verified seed {seed}: {len(records)} configurations so far", flush=True)
    report(records)
    print(f"PASS: activation sharing model {len(records)} configurations; scalar outputs verified")


if __name__ == "__main__":
    main()
