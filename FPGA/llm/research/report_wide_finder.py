"""Record completed wide-finder RTL tests and isolated synthesis measurements."""
from pathlib import Path
import csv
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return path.read_text(encoding="utf-8", errors="replace")


def main():
    unit = re.search(r"PASS: wide finder frames=(\d+) chunks=(\d+) cycles=(\d+) stalls=(\d+) replacements=(\d+)",
                     read(ROOT / "build/wide_finder/xsim.log"))
    end = re.search(r"PASS: wide sparse dot8 real DSP48E1 checked=(\d+) cycles=(\d+)",
                    read(ROOT / "build/wide_sparse_dot/xsim.log"))
    if not unit or not end:
        raise SystemExit("Missing successful simulation markers")
    raw = ROOT / "build/wide_sparse_dot/wide_performance.csv"
    with raw.open(newline="") as source:
        rows = list(csv.DictReader(source))
    table = {(r["case"], int(r["mode"]), int(r["limited_supply"])): r for r in rows}
    cases = ["dense_w8", "half_w8", "ten_percent_w8", "point_one_x_99", "zero_jobs", "short_jobs"]
    for case in cases:
        group = [table[case, m, limited] for m in range(5) for limited in range(2)]
        if len({r["source_slots"] for r in group}) != 1 or len({r["useful_products"] for r in group}) != 1:
            raise SystemExit(f"Mismatched work in {case}")
    # Anchor the new one-bank implementation and dense control to existing results.
    with (ROOT / "research/SPARSE_DOT8_PERFORMANCE.csv").open(newline="") as source:
        prior = {(r["case"], int(r["mode"])): r for r in csv.DictReader(source)}
    for case in cases:
        for new_mode, old_mode in [(0, 2), (4, 0)]:
            for key in ["source_slots", "useful_products", "command_to_result_cycles"]:
                if table[case, new_mode, 0][key] != prior[case, old_mode][key]:
                    raise SystemExit(f"Baseline changed: {case}, {key}")
    lines = ["# Wide finder measured results", "",
             "Generated from completed XSim and isolated synthesis runs. All widths retain eight DSP multipliers.", "",
             f"Standalone finder: {int(unit[1]):,} accepted frames / {int(unit[3]):,} cycles, with {int(unit[4]):,} output-stall cycles and {int(unit[5]):,} simultaneous frame replacements.",
             f"Real-DSP compute comparison: **{end[1]} result vectors / {sum(int(r['declared_jobs']) for r in rows)} declared dot results / {int(end[2]):,} cycles**, all passing.",
             "The scan-8 and dense anchors match the previous experiment on all main comparison fixtures.", ""]
    for limited, title in [(0, "Resident wide-word supply"), (1, "Supply limited to eight operand pairs per cycle")]:
        lines += [f"## {title}", "", "Command-to-result cycle counts (lower is better).", "",
                  "| Fixture | Useful products | Dense-8 | Finder-8 | Finder-16 | Finder-32 | Finder-64 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for case in cases:
            values = [table[case, m, limited]["command_to_result_cycles"] for m in [4, 0, 1, 2, 3]]
            lines.append(f"| {case} | {table[case, 0, limited]['useful_products']} | " + " | ".join(values) + " |")
        lines.append("")
    lines += ["## Useful utilization with resident input", "",
              "Useful products divided by eight times the complete command-to-result interval; this includes scanning and drain time.", "",
              "| Fixture | Finder-8 | Finder-16 | Finder-32 | Finder-64 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for case in ["half_w8", "ten_percent_w8", "point_one_x_99"]:
        vals = [100*int(table[case,m,0]["useful_products"])/(8*int(table[case,m,0]["command_to_result_cycles"])) for m in range(4)]
        lines.append(f"| {case} | " + " | ".join(f"{v:.3f}%" for v in vals) + " |")
    ratio = int(table["point_one_x_99",0,0]["command_to_result_cycles"])/int(table["point_one_x_99",3,0]["command_to_result_cycles"])
    lines += ["", f"The 64-position finder uses **{ratio:.2f}x fewer cycles** than the 8-position finder on the extreme resident fixture. With the eight-pair supply cap, that advantage disappears.",
              "Random 50% density rarely leaves a completely empty eight-position bank. One-bank-per-cycle arbitration limits the gain even when input words are wide.", "",
              "## Synthesis cost", "",
              "| Scan positions | LUTs | Registers | DSPs | BRAM | Setup WNS (ns) | Hold WHS (ns) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for width in [8,16,32,64]:
        folder=ROOT/f"build/wide_sparse_dot_synth/scan{width}"
        util=read(folder/"utilization.rpt"); timing=read(folder/"timing_synth.rpt")
        def count(pattern):
            return int(re.search(pattern,util).group(1))
        lut=count(r"\| Slice LUTs\*\s*\|\s*(\d+)")
        ff=count(r"\| Slice Registers\s*\|\s*(\d+)")
        dsp=count(r"\| DSPs\s*\|\s*(\d+)")
        bram=count(r"\| Block RAM Tile\s*\|\s*(\d+)")
        if dsp!=8: raise SystemExit("Arithmetic resource count changed")
        setup=re.search(r"Slack \([^)]+\)\s*:\s*([-\d.]+)ns\s*\(required time - arrival time\)",timing).group(1)
        hold=re.search(r"Slack \([^)]+\)\s*:\s*([-\d.]+)ns\s*\(arrival time - required time\)",timing).group(1)
        lines.append(f"| {width} | {lut} | {ff} | {dsp} | {bram} | {setup} | {hold} |")
    lines += ["", "Post-synthesis estimates at a 10 ns clock target with 0.300 ns uncertainty, xc7z020clg400-1. No place/route, external IO timing, physical clock location, memory banking, DDR or board measurements are included. Positive setup slack is not timing closure.",
              "", "The resident-width comparison increases available input-word width along with finder width. The capped-supply comparison charges acquisition from the first frame onward. The W4 stall fixture is correctness-only because its injected source bubbles are per frame, not per original position.",
              "", "[Architecture and reproduction](WIDE_FINDER.md) · [Raw metrics](WIDE_FINDER_PERFORMANCE.csv)", ""]
    target=ROOT/"research/WIDE_FINDER_RESULTS.md"
    target.write_text("\n".join(lines),encoding="utf-8")
    shutil.copyfile(raw,ROOT/"research/WIDE_FINDER_PERFORMANCE.csv")
    print(target)


if __name__=="__main__":
    main()
