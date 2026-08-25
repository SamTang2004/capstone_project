from __future__ import annotations

from dataclasses import dataclass
import struct

from .dtypes import OpType

DIAGNOSTICS_MAGIC = 0x47444342
DIAGNOSTICS_VERSION = 5
DIAG_TIMING, DIAG_MEMORY, DIAG_OPERATIONS, DIAG_UTILIZATION = 1, 2, 4, 8
DIAG_ALL = 15
DIAGNOSTICS_HEADER_STRUCT = struct.Struct("<IIIIIIIIQQ")
DIAGNOSTICS_RECORD_STRUCT = struct.Struct("<IIIIQQQQQQQII15QII")


@dataclass(frozen=True)
class DiagnosticsSettings:
    timing: bool = True
    memory: bool = True
    operations: bool = True
    utilization: bool = True
    clock_hz: int = 100_000_000
    peak_operations_per_cycle: int = 256  # two 8x8 arrays, two operations per MAC

    def __post_init__(self):
        if self.clock_hz <= 0 or self.peak_operations_per_cycle <= 0:
            raise ValueError("Diagnostics rates must be positive")
        if self.utilization and not (self.timing and self.operations):
            raise ValueError("Utilization requires timing and operations")

    @property
    def flags(self):
        return (self.timing * 1) | (self.memory * 2) | (self.operations * 4) | (self.utilization * 8)

    def initial_buffer(self, layers: int) -> bytes:
        if layers <= 0: raise ValueError("Diagnostics require model layers")
        header = DIAGNOSTICS_HEADER_STRUCT.pack(DIAGNOSTICS_MAGIC, DIAGNOSTICS_VERSION, self.flags, layers, 0, DIAGNOSTICS_RECORD_STRUCT.size, self.clock_hz, self.peak_operations_per_cycle, 0, 0)
        return header + bytes(layers * DIAGNOSTICS_RECORD_STRUCT.size)


@dataclass(frozen=True)
class LayerDiagnostics:
    layer_index: int; op_type: int; status: int; elapsed_ns: int; cycles: int
    input_bytes: int; output_bytes: int; weight_bytes: int; operations: int
    throughput_ops_per_second: int; utilization_percent: float
    gemm_bias_elapsed_ns: int = 0
    hardware_counters: tuple[int, ...] = ()

    @property
    def op_name(self):
        try: return OpType(self.op_type).name
        except ValueError: return f"OP_{self.op_type}"

    @property
    def transformer_cycles(self):
        names = (
            "window_cycles", "busy_cycles", "memory_stall_cycles",
            "input_stall_cycles", "output_stall_cycles", "last_op_cycles",
            "max_op_cycles", "gemm_load_a_cycles", "gemm_load_b_cycles",
            "gemm_load_b_weight_cycles", "gemm_load_b_bias_cycles",
            "gemm_w4_cache_hit_tiles", "gemm_w4_cache_miss_tiles",
            "gemm_compute_cycles", "gemm_write_cycles", "completed_ops",
            "hardware_status",
        )
        return dict(zip(names, self.hardware_counters))

    @property
    def hardware_window_cycles(self) -> int:
        return self.hardware_counters[0] if self.hardware_counters else 0

    @property
    def hardware_busy_cycles(self) -> int:
        return self.hardware_counters[1] if len(self.hardware_counters) > 1 else 0

    @property
    def hardware_completed_ops(self) -> int:
        return self.hardware_counters[15] if len(self.hardware_counters) > 15 else 0


@dataclass(frozen=True)
class RunDiagnostics:
    settings: DiagnosticsSettings
    layers: list[LayerDiagnostics]
    firmware_total_ns: int
    arena_bytes_used: int
    host_write_ns: int = 0
    host_execute_ns: int = 0
    host_read_ns: int = 0
    host_total_ns: int = 0


@dataclass(frozen=True)
class DiagnosticResult:
    output: object
    diagnostics: RunDiagnostics


def parse_diagnostics(raw, settings: DiagnosticsSettings) -> RunDiagnostics:
    raw = raw.tobytes() if hasattr(raw, "tobytes") else bytes(raw)
    fields = DIAGNOSTICS_HEADER_STRUCT.unpack_from(raw)
    magic, version, flags, capacity, count, record_bytes, clock_hz, peak, total_ns, arena = fields
    if (magic, version, flags, record_bytes, clock_hz, peak) != (DIAGNOSTICS_MAGIC, DIAGNOSTICS_VERSION, settings.flags, DIAGNOSTICS_RECORD_STRUCT.size, settings.clock_hz, settings.peak_operations_per_cycle):
        raise ValueError("Incompatible diagnostics buffer")
    if count > capacity or DIAGNOSTICS_HEADER_STRUCT.size + count * record_bytes > len(raw):
        raise ValueError("Invalid diagnostics record count")
    layers = []
    for i in range(count):
        record = DIAGNOSTICS_RECORD_STRUCT.unpack_from(raw, DIAGNOSTICS_HEADER_STRUCT.size + i * record_bytes)
        layers.append(LayerDiagnostics(
            record[0], record[1], record[2], record[4], record[5], record[6],
            record[7], record[8], record[9], record[10], record[11] / 10000.0,
            record[12], tuple(record[13:])))
    return RunDiagnostics(settings, layers, total_ns, arena)


def summarize_cycle_report(report: RunDiagnostics, layer_names=None, layer_stages=None) -> dict:
    """Return JSON-safe totals, operation/stage groups, and exact per-layer counters."""
    names = list(layer_names or ())
    stages = list(layer_stages or ())
    if names and len(names) != len(report.layers):
        raise ValueError("Layer-name count does not match diagnostics")
    if stages and len(stages) != len(report.layers):
        raise ValueError("Layer-stage count does not match diagnostics")

    additive_counters = (
        "window_cycles", "busy_cycles", "memory_stall_cycles",
        "input_stall_cycles", "output_stall_cycles", "completed_ops",
        "gemm_load_a_cycles", "gemm_load_b_cycles",
        "gemm_load_b_weight_cycles", "gemm_load_b_bias_cycles",
        "gemm_w4_cache_hit_tiles", "gemm_w4_cache_miss_tiles",
        "gemm_compute_cycles", "gemm_write_cycles",
    )

    def empty_group():
        return {
            "layer_count": 0,
            "firmware_elapsed_ns": 0,
            "gemm_bias_elapsed_ns": 0,
            "operations": 0,
            **{key: 0 for key in additive_counters},
            "hardware_op_cycles": 0,
            "largest_op_cycles": 0,
            "hardware_status_or": 0,
        }

    totals = empty_group()
    by_operation = {}
    by_stage = {}
    per_layer = []
    for position, layer in enumerate(report.layers):
        counters = layer.transformer_cycles
        stage = stages[position] if stages else "full_model"
        name = names[position] if names else f"layer_{layer.layer_index}.{layer.op_name}"
        groups = (
            totals,
            by_operation.setdefault(layer.op_name, empty_group()),
            by_stage.setdefault(stage, empty_group()),
        )
        for group in groups:
            group["layer_count"] += 1
            group["firmware_elapsed_ns"] += layer.elapsed_ns
            group["gemm_bias_elapsed_ns"] += layer.gemm_bias_elapsed_ns
            group["operations"] += layer.operations
            for key in additive_counters:
                value = counters.get(key, 0)
                group[key] += value
            if counters.get("completed_ops", 0):
                group["hardware_op_cycles"] += counters.get("last_op_cycles", 0)
            group["largest_op_cycles"] = max(
                group["largest_op_cycles"], counters.get("max_op_cycles", 0))
            group["hardware_status_or"] |= counters.get("hardware_status", 0)
        per_layer.append({
            "layer_index": layer.layer_index,
            "name": name,
            "stage": stage,
            "operation": layer.op_name,
            "status": layer.status,
            "firmware_elapsed_ns": layer.elapsed_ns,
            "operations": layer.operations,
            "input_bytes": layer.input_bytes,
            "output_bytes": layer.output_bytes,
            "weight_bytes": layer.weight_bytes,
            "gemm_bias_elapsed_ns": layer.gemm_bias_elapsed_ns,
            **counters,
        })

    clock_hz = report.settings.clock_hz
    for group in [totals, *by_operation.values(), *by_stage.values()]:
        group["window_seconds"] = group.get("window_cycles", 0) / clock_hz
        group["busy_seconds"] = group.get("busy_cycles", 0) / clock_hz
        window = group.get("window_cycles", 0)
        group["busy_percent"] = 100.0 * group.get("busy_cycles", 0) / window if window else 0.0
        group["memory_stall_percent"] = 100.0 * group.get("memory_stall_cycles", 0) / window if window else 0.0
        group["effective_gops"] = group["operations"] / group["firmware_elapsed_ns"] if group["firmware_elapsed_ns"] else 0.0
        group["gemm_bias_cycles"] = group["gemm_bias_elapsed_ns"] * clock_hz // 1_000_000_000

    def ranked_breakdown(groups, label):
        total_cycles = totals["window_cycles"]
        return [
            {
                label: name,
                **group,
                "total_window_percent": (
                    100.0 * group["window_cycles"] / total_cycles
                    if total_cycles else 0.0
                ),
            }
            for name, group in sorted(
                groups.items(), key=lambda item: item[1]["window_cycles"], reverse=True)
        ]

    top_layers = sorted(per_layer, key=lambda item: item["window_cycles"], reverse=True)[:25]
    return {
        "clock_hz": clock_hz,
        "peak_operations_per_cycle": report.settings.peak_operations_per_cycle,
        "firmware_total_ns": report.firmware_total_ns,
        "host_write_ns": report.host_write_ns,
        "host_execute_ns": report.host_execute_ns,
        "host_read_ns": report.host_read_ns,
        "host_total_ns": report.host_total_ns,
        "arena_bytes_used": report.arena_bytes_used,
        "totals": totals,
        "by_operation": by_operation,
        "by_stage": by_stage,
        # Ranked tables are included in every diagnostic report so performance
        # tests always expose where the complete device window was consumed.
        "module_cycle_breakdown": ranked_breakdown(by_operation, "module"),
        "stage_cycle_breakdown": ranked_breakdown(by_stage, "stage"),
        "top_layers_by_window_cycles": top_layers,
        "layers": per_layer,
    }
