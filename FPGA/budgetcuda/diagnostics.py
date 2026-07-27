from __future__ import annotations

import csv
from dataclasses import dataclass
import io
from pathlib import Path
import struct

from .dtypes import OpType


DIAGNOSTICS_MAGIC = 0x47444342
DIAGNOSTICS_VERSION = 2
DIAG_TIMING = 1 << 0
DIAG_MEMORY = 1 << 1
DIAG_OPERATIONS = 1 << 2
DIAG_UTILIZATION = 1 << 3
DIAG_HARDWARE = 1 << 4
DIAG_ACCURACY = 1 << 5
DIAG_ALL = (
    DIAG_TIMING | DIAG_MEMORY | DIAG_OPERATIONS | DIAG_UTILIZATION
    | DIAG_HARDWARE | DIAG_ACCURACY
)
DIAGNOSTICS_HEADER_STRUCT = struct.Struct("<IIIIIIIIQQ")
DIAGNOSTICS_RECORD_STRUCT = struct.Struct(
    "<IIIIQQQQQQQII" + "Q" * 14 + "QqQiiII"
)


@dataclass(frozen=True)
class DiagnosticsSettings:
    timing: bool = True
    memory: bool = True
    operations: bool = True
    utilization: bool = True
    hardware: bool = True
    accuracy: bool = True
    clock_hz: int = 100_000_000
    peak_operations_per_cycle: int = 512

    def __post_init__(self):
        if self.clock_hz <= 0 or self.peak_operations_per_cycle <= 0:
            raise ValueError("Diagnostics clock and peak operations per cycle must be positive")
        if self.utilization and not (self.timing and self.operations):
            raise ValueError("Utilization diagnostics require timing and operation counts")
        if self.flags == 0:
            raise ValueError("At least one diagnostics report must be enabled")

    @property
    def flags(self) -> int:
        return (
            (DIAG_TIMING if self.timing else 0)
            | (DIAG_MEMORY if self.memory else 0)
            | (DIAG_OPERATIONS if self.operations else 0)
            | (DIAG_UTILIZATION if self.utilization else 0)
            | (DIAG_HARDWARE if self.hardware else 0)
            | (DIAG_ACCURACY if self.accuracy else 0)
        )

    def buffer_bytes(self, layer_count: int) -> int:
        if layer_count <= 0:
            raise ValueError("Diagnostics require at least one model layer")
        return DIAGNOSTICS_HEADER_STRUCT.size + layer_count * DIAGNOSTICS_RECORD_STRUCT.size

    def initial_buffer(self, layer_count: int) -> bytes:
        size = self.buffer_bytes(layer_count)
        header = DIAGNOSTICS_HEADER_STRUCT.pack(
            DIAGNOSTICS_MAGIC,
            DIAGNOSTICS_VERSION,
            self.flags,
            layer_count,
            0,
            DIAGNOSTICS_RECORD_STRUCT.size,
            self.clock_hz,
            self.peak_operations_per_cycle,
            0,
            0,
        )
        return header + bytes(size - len(header))


@dataclass(frozen=True)
class LayerDiagnostics:
    layer_index: int
    op_type: int
    op_name: str
    status: int
    elapsed_ns: int
    cycles: int
    input_bytes: int
    output_bytes: int
    weight_bytes: int
    operations: int
    throughput_ops_per_second: int
    utilization_percent: float
    pl_total_cycles: int
    pl_engine_cycles: int
    pl_compute_cycles: int
    pl_load_cycles: int
    pl_cache_wait_cycles: int
    pl_cache_ready_cycles: int
    pl_ddr_read_commands: int
    pl_ddr_read_beats: int
    pl_ddr_write_commands: int
    pl_ddr_stall_cycles: int
    pl_core_rows: int
    pl_output_stall_cycles: int
    pl_descriptor_count: int
    pl_cache_full_cycles: int
    output_hash: int
    output_sum: int
    output_elements: int
    output_min: int
    output_max: int
    output_zero_count: int
    output_saturation_count: int

    @property
    def working_set_bytes(self) -> int:
        return self.input_bytes + self.output_bytes + self.weight_bytes

    @staticmethod
    def _percent(numerator: int, denominator: int) -> float:
        return 100.0 * numerator / denominator if denominator else 0.0

    @property
    def compute_duty_percent(self) -> float:
        return self._percent(self.pl_compute_cycles, self.pl_total_cycles)

    @property
    def engine_duty_percent(self) -> float:
        return self._percent(self.pl_engine_cycles, self.pl_total_cycles)

    @property
    def loader_starvation_percent(self) -> float:
        return self._percent(self.pl_cache_wait_cycles, self.pl_load_cycles)

    @property
    def ddr_stall_percent(self) -> float:
        return self._percent(self.pl_ddr_stall_cycles, self.pl_total_cycles)

    @property
    def output_stall_percent(self) -> float:
        return self._percent(self.pl_output_stall_cycles, self.pl_total_cycles)

    @property
    def zero_percent(self) -> float:
        return self._percent(self.output_zero_count, self.output_elements)

    @property
    def saturation_percent(self) -> float:
        return self._percent(
            self.output_saturation_count, self.output_elements
        )

    @property
    def likely_bottleneck(self) -> str:
        if not self.pl_total_cycles:
            return "no PL sample"
        candidates = {
            "DDR backpressure": self.pl_ddr_stall_cycles,
            "tile cache starvation": self.pl_cache_wait_cycles,
            "output backpressure": self.pl_output_stall_cycles,
            "compute": self.pl_compute_cycles,
        }
        return max(candidates, key=candidates.get)


@dataclass
class RunDiagnostics:
    settings: DiagnosticsSettings
    layers: list[LayerDiagnostics]
    firmware_total_ns: int
    arena_bytes_used: int
    host_write_ns: int = 0
    host_execute_ns: int = 0
    host_read_ns: int = 0
    host_total_ns: int = 0

    @property
    def time_curve(self) -> list[int]:
        return [layer.elapsed_ns for layer in self.layers]

    @property
    def memory_curve(self) -> list[int]:
        return [layer.working_set_bytes for layer in self.layers]

    @property
    def usage_curve(self) -> list[dict]:
        return [
            {
                "input_bytes": layer.input_bytes,
                "output_bytes": layer.output_bytes,
                "weight_bytes": layer.weight_bytes,
                "working_set_bytes": layer.working_set_bytes,
            }
            for layer in self.layers
        ]

    @property
    def utilization_curve(self) -> list[float]:
        return [layer.utilization_percent for layer in self.layers]

    @property
    def operation_curve(self) -> list[int]:
        return [layer.operations for layer in self.layers]

    @property
    def compute_duty_curve(self) -> list[float]:
        return [layer.compute_duty_percent for layer in self.layers]

    def first_divergence(
        self, reference: "RunDiagnostics"
    ) -> tuple[LayerDiagnostics, LayerDiagnostics] | None:
        for actual, expected in zip(self.layers, reference.layers):
            if (
                actual.output_hash
                and expected.output_hash
                and actual.output_hash != expected.output_hash
            ):
                return actual, expected
        if len(self.layers) != len(reference.layers):
            index = min(len(self.layers), len(reference.layers))
            if index < len(self.layers) and reference.layers:
                return self.layers[index], reference.layers[-1]
            if index < len(reference.layers) and self.layers:
                return self.layers[-1], reference.layers[index]
        return None

    def accuracy_mismatches(
        self, reference: "RunDiagnostics"
    ) -> list[dict]:
        mismatches = []
        for actual, expected in zip(self.layers, reference.layers):
            if actual.output_hash and expected.output_hash and (
                actual.output_hash != expected.output_hash
            ):
                mismatches.append({
                    "layer_index": actual.layer_index,
                    "op_name": actual.op_name,
                    "actual_hash": f"{actual.output_hash:016x}",
                    "expected_hash": f"{expected.output_hash:016x}",
                    "actual_range": (actual.output_min, actual.output_max),
                    "expected_range": (expected.output_min, expected.output_max),
                    "actual_saturation_percent":
                        actual.saturation_percent,
                    "expected_saturation_percent":
                        expected.saturation_percent,
                })
        return mismatches

    def _rows(self) -> list[dict]:
        return [
            {
                "layer_index": layer.layer_index,
                "op_type": layer.op_type,
                "op_name": layer.op_name,
                "status": layer.status,
                "elapsed_ns": layer.elapsed_ns,
                "cycles": layer.cycles,
                "input_bytes": layer.input_bytes,
                "output_bytes": layer.output_bytes,
                "weight_bytes": layer.weight_bytes,
                "working_set_bytes": layer.working_set_bytes,
                "operations": layer.operations,
                "throughput_ops_per_second": layer.throughput_ops_per_second,
                "utilization_percent": f"{layer.utilization_percent:.6f}",
                "pl_total_cycles": layer.pl_total_cycles,
                "pl_engine_cycles": layer.pl_engine_cycles,
                "pl_compute_cycles": layer.pl_compute_cycles,
                "pl_load_cycles": layer.pl_load_cycles,
                "pl_cache_wait_cycles": layer.pl_cache_wait_cycles,
                "pl_cache_ready_cycles": layer.pl_cache_ready_cycles,
                "pl_ddr_read_commands": layer.pl_ddr_read_commands,
                "pl_ddr_read_beats": layer.pl_ddr_read_beats,
                "pl_ddr_write_commands": layer.pl_ddr_write_commands,
                "pl_ddr_stall_cycles": layer.pl_ddr_stall_cycles,
                "pl_core_rows": layer.pl_core_rows,
                "pl_output_stall_cycles": layer.pl_output_stall_cycles,
                "pl_descriptor_count": layer.pl_descriptor_count,
                "pl_cache_full_cycles": layer.pl_cache_full_cycles,
                "compute_duty_percent": f"{layer.compute_duty_percent:.6f}",
                "loader_starvation_percent":
                    f"{layer.loader_starvation_percent:.6f}",
                "ddr_stall_percent": f"{layer.ddr_stall_percent:.6f}",
                "likely_bottleneck": layer.likely_bottleneck,
                "output_hash": f"{layer.output_hash:016x}",
                "output_sum": layer.output_sum,
                "output_elements": layer.output_elements,
                "output_min": layer.output_min,
                "output_max": layer.output_max,
                "output_zero_count": layer.output_zero_count,
                "output_saturation_count": layer.output_saturation_count,
                "zero_percent": f"{layer.zero_percent:.6f}",
                "saturation_percent": f"{layer.saturation_percent:.6f}",
            }
            for layer in self.layers
        ]

    def to_csv(self) -> str:
        fields = [
            "layer_index", "op_type", "op_name", "status", "elapsed_ns", "cycles",
            "input_bytes", "output_bytes", "weight_bytes", "working_set_bytes",
            "operations", "throughput_ops_per_second", "utilization_percent",
            "pl_total_cycles", "pl_engine_cycles", "pl_compute_cycles",
            "pl_load_cycles", "pl_cache_wait_cycles", "pl_cache_ready_cycles",
            "pl_ddr_read_commands", "pl_ddr_read_beats",
            "pl_ddr_write_commands", "pl_ddr_stall_cycles", "pl_core_rows",
            "pl_output_stall_cycles", "pl_descriptor_count",
            "pl_cache_full_cycles", "compute_duty_percent",
            "loader_starvation_percent", "ddr_stall_percent",
            "likely_bottleneck", "output_hash", "output_sum",
            "output_elements", "output_min", "output_max",
            "output_zero_count", "output_saturation_count", "zero_percent",
            "saturation_percent",
        ]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(self._rows())
        return output.getvalue()

    def to_markdown(self) -> str:
        lines = [
            "# BudgetCUDA Run Diagnostics",
            "",
            f"- Firmware layer time: {self.firmware_total_ns} ns",
            f"- Host total time: {self.host_total_ns} ns",
            f"- Firmware arena high-water usage: {self.arena_bytes_used} bytes",
            "",
            "| Layer | Operation | Time (ns) | PL cycles | Compute duty | Loader starve | DDR stall | Bottleneck | Hash | Saturation |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
        ]
        for layer in self.layers:
            lines.append(
                f"| {layer.layer_index} | {layer.op_name} | {layer.elapsed_ns} | "
                f"{layer.pl_total_cycles} | {layer.compute_duty_percent:.3f}% | "
                f"{layer.loader_starvation_percent:.3f}% | "
                f"{layer.ddr_stall_percent:.3f}% | {layer.likely_bottleneck} | "
                f"`{layer.output_hash:016x}` | "
                f"{layer.saturation_percent:.3f}% |"
            )
        return "\n".join(lines) + "\n"

    def to_text(self) -> str:
        lines = [
            "BudgetCUDA Run Diagnostics",
            f"firmware_total_ns={self.firmware_total_ns}",
            f"host_total_ns={self.host_total_ns}",
            f"arena_bytes_used={self.arena_bytes_used}",
        ]
        lines.extend(
            f"layer={layer.layer_index} op={layer.op_name} status={layer.status} "
            f"elapsed_ns={layer.elapsed_ns} cycles={layer.cycles} "
            f"input_bytes={layer.input_bytes} output_bytes={layer.output_bytes} "
            f"weight_bytes={layer.weight_bytes} operations={layer.operations} "
            f"throughput_ops_per_second={layer.throughput_ops_per_second} "
            f"roofline_utilization_percent={layer.utilization_percent:.3f} "
            f"pl_total_cycles={layer.pl_total_cycles} "
            f"compute_duty_percent={layer.compute_duty_percent:.3f} "
            f"loader_starvation_percent={layer.loader_starvation_percent:.3f} "
            f"ddr_stall_percent={layer.ddr_stall_percent:.3f} "
            f"bottleneck={layer.likely_bottleneck!r} "
            f"output_hash={layer.output_hash:016x} "
            f"output_range=[{layer.output_min},{layer.output_max}] "
            f"saturation_percent={layer.saturation_percent:.3f}"
            for layer in self.layers
        )
        return "\n".join(lines) + "\n"

    def to_txt(self) -> str:
        return self.to_text()

    def export(self, destination, format: str | None = None) -> Path:
        path = Path(destination)
        selected = (format or path.suffix.lstrip(".") or "txt").lower()
        if selected == "csv":
            content = self.to_csv()
        elif selected in ("md", "markdown"):
            content = self.to_markdown()
        elif selected in ("txt", "text"):
            content = self.to_text()
        else:
            raise ValueError(f"Unsupported diagnostics export format: {selected}")
        path.write_text(content, encoding="utf-8")
        return path

    def export_csv(self, destination) -> Path:
        return self.export(destination, "csv")

    def export_markdown(self, destination) -> Path:
        return self.export(destination, "md")

    def export_text(self, destination) -> Path:
        return self.export(destination, "txt")


@dataclass(frozen=True)
class DiagnosticResult:
    output: object
    diagnostics: RunDiagnostics


def parse_diagnostics(raw, settings: DiagnosticsSettings) -> RunDiagnostics:
    if hasattr(raw, "tobytes"):
        raw = raw.tobytes()
    elif not isinstance(raw, (bytes, bytearray, memoryview)):
        raw = bytes(int(value) & 0xFF for value in raw)
    raw = bytes(raw)
    if len(raw) < DIAGNOSTICS_HEADER_STRUCT.size:
        raise ValueError("Diagnostics buffer is shorter than its header")
    header = DIAGNOSTICS_HEADER_STRUCT.unpack_from(raw)
    magic, version, flags, capacity, layer_count, record_bytes, clock_hz, peak_ops, total_ns, arena_bytes = header
    if magic != DIAGNOSTICS_MAGIC or version != DIAGNOSTICS_VERSION:
        raise ValueError("Firmware returned an incompatible diagnostics buffer")
    if flags != settings.flags or clock_hz != settings.clock_hz or peak_ops != settings.peak_operations_per_cycle:
        raise ValueError("Firmware diagnostics settings do not match the requested settings")
    if record_bytes != DIAGNOSTICS_RECORD_STRUCT.size or layer_count != capacity:
        raise ValueError("Firmware returned invalid diagnostics record metadata")
    if DIAGNOSTICS_HEADER_STRUCT.size + layer_count * record_bytes > len(raw):
        raise ValueError("Firmware diagnostics records exceed their buffer")
    layers = []
    for index in range(layer_count):
        fields = DIAGNOSTICS_RECORD_STRUCT.unpack_from(
            raw, DIAGNOSTICS_HEADER_STRUCT.size + index * record_bytes
        )
        (
            layer_index, op_type, status, _, elapsed_ns, cycles, input_bytes,
            output_bytes, weight_bytes, operations, throughput,
            utilization_ppm, _, *extension,
        ) = fields
        try:
            op_name = OpType(op_type).name
        except ValueError:
            op_name = f"OP_{op_type}"
        layers.append(
            LayerDiagnostics(
                layer_index, op_type, op_name, status, elapsed_ns, cycles,
                input_bytes, output_bytes, weight_bytes, operations, throughput,
                utilization_ppm / 10_000.0, *extension,
            )
        )
    return RunDiagnostics(settings, layers, total_ns, arena_bytes)
