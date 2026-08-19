from __future__ import annotations

import csv
from dataclasses import dataclass, field
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
DIAG_ALL = DIAG_TIMING | DIAG_MEMORY | DIAG_OPERATIONS | DIAG_UTILIZATION

DIAGNOSTICS_HEADER_STRUCT = struct.Struct("<IIIIIIIIQQ")
DIAGNOSTICS_RECORD_STRUCT = struct.Struct("<IIIIQQQQQQQII16I")


@dataclass(frozen=True)
class DiagnosticsSettings:
    """Select firmware and host metrics for one model run."""

    timing: bool = True
    memory: bool = True
    operations: bool = True
    utilization: bool = True
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
    conv_cycles_total: int = 0
    conv_cycles_zero_a: int = 0
    conv_cycles_load_a: int = 0
    conv_cycles_zero_b: int = 0
    conv_cycles_load_b: int = 0
    conv_cycles_bias: int = 0
    conv_cycles_compute: int = 0
    conv_cycles_accumulate: int = 0
    conv_cycles_write: int = 0
    conv_loader_address_cycles: int = 0
    conv_loader_request_cycles: int = 0
    conv_loader_receive_cycles: int = 0
    conv_cache_hits: int = 0
    conv_cache_misses: int = 0
    conv_fill_cycles: int = 0
    conv_bytes_delivered: int = 0

    @property
    def working_set_bytes(self) -> int:
        return self.input_bytes + self.output_bytes + self.weight_bytes

    @property
    def conv_cycle_breakdown(self) -> dict[str, int]:
        return {
            "zero_a": self.conv_cycles_zero_a,
            "load_a": self.conv_cycles_load_a,
            "zero_b": self.conv_cycles_zero_b,
            "load_b": self.conv_cycles_load_b,
            "bias": self.conv_cycles_bias,
            "compute": self.conv_cycles_compute,
            "accumulate": self.conv_cycles_accumulate,
            "write": self.conv_cycles_write,
            "controller_other": max(0, self.conv_cycles_total - sum((
                self.conv_cycles_zero_a, self.conv_cycles_load_a, self.conv_cycles_zero_b,
                self.conv_cycles_load_b, self.conv_cycles_bias, self.conv_cycles_compute,
                self.conv_cycles_accumulate, self.conv_cycles_write,
            ))),
        }


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

    def to_csv(self) -> str:
        output = io.StringIO()
        fields = list(self._rows()[0]) if self.layers else [
            "layer_index", "op_type", "op_name", "status", "elapsed_ns", "cycles",
            "input_bytes", "output_bytes", "weight_bytes", "working_set_bytes",
            "operations", "throughput_ops_per_second", "utilization_percent",
        ]
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
            f"- Host input write: {self.host_write_ns} ns",
            f"- Host execute round trip: {self.host_execute_ns} ns",
            f"- Host output read: {self.host_read_ns} ns",
            f"- Firmware arena high-water usage: {self.arena_bytes_used} bytes",
            "",
            "| Layer | Operation | Status | Time (ns) | Cycles | Input B | Output B | Weight B | Operations | Ops/s | Utilization |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for layer in self.layers:
            lines.append(
                f"| {layer.layer_index} | {layer.op_name} | {layer.status} | {layer.elapsed_ns} | "
                f"{layer.cycles} | {layer.input_bytes} | {layer.output_bytes} | "
                f"{layer.weight_bytes} | {layer.operations} | {layer.throughput_ops_per_second} | "
                f"{layer.utilization_percent:.3f}% |"
            )
        conv_layers = [layer for layer in self.layers if layer.conv_cycles_total]
        if conv_layers:
            lines.extend([
                "", "## Conv2D Hardware Cycle Breakdown", "",
                "| Layer | Total | Zero A | Load A | Zero B | Load B | Bias | Compute | Accumulate | Write | Other | Loader address | Loader request | Loader receive | Cache hits | Cache misses | Fill cycles | Bytes delivered |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ])
            for layer in conv_layers:
                phases = layer.conv_cycle_breakdown
                lines.append(
                    f"| {layer.layer_index} | {layer.conv_cycles_total} | {phases['zero_a']} | {phases['load_a']} | "
                    f"{phases['zero_b']} | {phases['load_b']} | {phases['bias']} | {phases['compute']} | "
                    f"{phases['accumulate']} | {phases['write']} | {phases['controller_other']} | "
                    f"{layer.conv_loader_address_cycles} | {layer.conv_loader_request_cycles} | "
                    f"{layer.conv_loader_receive_cycles} | {layer.conv_cache_hits} | {layer.conv_cache_misses} | "
                    f"{layer.conv_fill_cycles} | {layer.conv_bytes_delivered} |"
                )
        return "\n".join(lines) + "\n"

    def to_text(self) -> str:
        lines = [
            "BudgetCUDA Run Diagnostics",
            f"firmware_total_ns={self.firmware_total_ns}",
            f"host_total_ns={self.host_total_ns}",
            f"host_write_ns={self.host_write_ns}",
            f"host_execute_ns={self.host_execute_ns}",
            f"host_read_ns={self.host_read_ns}",
            f"arena_bytes_used={self.arena_bytes_used}",
        ]
        for layer in self.layers:
            lines.append(
                f"layer={layer.layer_index} op={layer.op_name} status={layer.status} "
                f"elapsed_ns={layer.elapsed_ns} cycles={layer.cycles} "
                f"input_bytes={layer.input_bytes} output_bytes={layer.output_bytes} "
                f"weight_bytes={layer.weight_bytes} operations={layer.operations} "
                f"throughput_ops_per_second={layer.throughput_ops_per_second} "
                f"utilization_percent={layer.utilization_percent:.3f}"
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
                "conv_cycles_total": layer.conv_cycles_total,
                "conv_cycles_zero_a": layer.conv_cycles_zero_a,
                "conv_cycles_load_a": layer.conv_cycles_load_a,
                "conv_cycles_zero_b": layer.conv_cycles_zero_b,
                "conv_cycles_load_b": layer.conv_cycles_load_b,
                "conv_cycles_bias": layer.conv_cycles_bias,
                "conv_cycles_compute": layer.conv_cycles_compute,
                "conv_cycles_accumulate": layer.conv_cycles_accumulate,
                "conv_cycles_write": layer.conv_cycles_write,
                "conv_controller_other_cycles": layer.conv_cycle_breakdown["controller_other"],
                "conv_loader_address_cycles": layer.conv_loader_address_cycles,
                "conv_loader_request_cycles": layer.conv_loader_request_cycles,
                "conv_loader_receive_cycles": layer.conv_loader_receive_cycles,
                "conv_cache_hits": layer.conv_cache_hits,
                "conv_cache_misses": layer.conv_cache_misses,
                "conv_fill_cycles": layer.conv_fill_cycles,
                "conv_bytes_delivered": layer.conv_bytes_delivered,
            }
            for layer in self.layers
        ]


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
    (
        magic,
        version,
        flags,
        capacity,
        layer_count,
        record_bytes,
        clock_hz,
        peak_operations_per_cycle,
        total_elapsed_ns,
        arena_bytes_used,
    ) = DIAGNOSTICS_HEADER_STRUCT.unpack_from(raw)
    if magic != DIAGNOSTICS_MAGIC or version != DIAGNOSTICS_VERSION:
        raise ValueError("Firmware returned an incompatible diagnostics buffer")
    if flags != settings.flags or clock_hz != settings.clock_hz or peak_operations_per_cycle != settings.peak_operations_per_cycle:
        raise ValueError("Firmware diagnostics settings do not match the requested settings")
    if record_bytes != DIAGNOSTICS_RECORD_STRUCT.size or layer_count > capacity:
        raise ValueError("Firmware returned invalid diagnostics record metadata")
    if layer_count != capacity:
        raise ValueError(
            f"Firmware returned {layer_count} diagnostics records for {capacity} layers; "
            "the daemon may not support this diagnostics ABI"
        )
    required = DIAGNOSTICS_HEADER_STRUCT.size + layer_count * record_bytes
    if required > len(raw):
        raise ValueError("Firmware diagnostics records exceed their buffer")

    layers = []
    for index in range(layer_count):
        fields = DIAGNOSTICS_RECORD_STRUCT.unpack_from(
            raw, DIAGNOSTICS_HEADER_STRUCT.size + index * record_bytes
        )
        (
            layer_index,
            op_type,
            status,
            _,
            elapsed_ns,
            cycles,
            input_bytes,
            output_bytes,
            weight_bytes,
            operations,
            throughput,
            utilization_ppm,
            _,
            *conv_profile,
        ) = fields
        try:
            op_name = OpType(op_type).name
        except ValueError:
            op_name = f"OP_{op_type}"
        layers.append(
            LayerDiagnostics(
                layer_index,
                op_type,
                op_name,
                status,
                elapsed_ns,
                cycles,
                input_bytes,
                output_bytes,
                weight_bytes,
                operations,
                throughput,
                utilization_ppm / 10_000.0,
                *conv_profile,
            )
        )
    return RunDiagnostics(settings, layers, total_elapsed_ns, arena_bytes_used)
