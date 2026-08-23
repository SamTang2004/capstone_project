from __future__ import annotations

from dataclasses import dataclass
import struct

from .dtypes import OpType

DIAGNOSTICS_MAGIC = 0x47444342
DIAGNOSTICS_VERSION = 2
DIAG_TIMING, DIAG_MEMORY, DIAG_OPERATIONS, DIAG_UTILIZATION = 1, 2, 4, 8
DIAG_ALL = 15
DIAGNOSTICS_HEADER_STRUCT = struct.Struct("<IIIIIIIIQQ")
DIAGNOSTICS_RECORD_STRUCT = struct.Struct("<IIIIQQQQQQQII16I")


@dataclass(frozen=True)
class DiagnosticsSettings:
    timing: bool = True
    memory: bool = True
    operations: bool = True
    utilization: bool = True
    clock_hz: int = 100_000_000
    peak_operations_per_cycle: int = 128  # dual 8x8 INT8 MAC arrays

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
        header = DIAGNOSTICS_HEADER_STRUCT.pack(DIAGNOSTICS_MAGIC, 2, self.flags, layers, 0, DIAGNOSTICS_RECORD_STRUCT.size, self.clock_hz, self.peak_operations_per_cycle, 0, 0)
        return header + bytes(layers * DIAGNOSTICS_RECORD_STRUCT.size)


@dataclass(frozen=True)
class LayerDiagnostics:
    layer_index: int; op_type: int; status: int; elapsed_ns: int; cycles: int
    input_bytes: int; output_bytes: int; weight_bytes: int; operations: int
    throughput_ops_per_second: int; utilization_percent: float
    hardware_counters: tuple[int, ...] = ()

    @property
    def op_name(self):
        try: return OpType(self.op_type).name
        except ValueError: return f"OP_{self.op_type}"

    @property
    def transformer_cycles(self):
        names = ("window_cycles", "busy_cycles", "memory_stall_cycles", "input_stall_cycles", "output_stall_cycles", "last_op_cycles", "max_op_cycles", "completed_ops", "active_op", "error_latched", "trace_valid", "trace_event", "trace_data", "reserved0", "reserved1", "reserved2")
        return dict(zip(names, self.hardware_counters))


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
    if (magic, version, flags, record_bytes, clock_hz, peak) != (DIAGNOSTICS_MAGIC, 2, settings.flags, DIAGNOSTICS_RECORD_STRUCT.size, settings.clock_hz, settings.peak_operations_per_cycle):
        raise ValueError("Incompatible diagnostics buffer")
    if count > capacity or DIAGNOSTICS_HEADER_STRUCT.size + count * record_bytes > len(raw):
        raise ValueError("Invalid diagnostics record count")
    layers = []
    for i in range(count):
        record = DIAGNOSTICS_RECORD_STRUCT.unpack_from(raw, DIAGNOSTICS_HEADER_STRUCT.size + i * record_bytes)
        layers.append(LayerDiagnostics(record[0], record[1], record[2], record[4], record[5], record[6], record[7], record[8], record[9], record[10], record[11] / 10000.0, tuple(record[13:])))
    return RunDiagnostics(settings, layers, total_ns, arena)
