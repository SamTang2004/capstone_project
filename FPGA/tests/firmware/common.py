from __future__ import annotations

import ctypes
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIRMWARE = ROOT / "firmware"
INCLUDE = FIRMWARE / "include"
SRC = FIRMWARE / "src"

BC_STATUS_OK = 0
BC_STATUS_GENERAL_ERROR = 1
BC_STATUS_INVALID_COMMAND = 2
BC_STATUS_INVALID_HANDLE = 3
BC_STATUS_OUT_OF_MEMORY = 4
BC_STATUS_SHAPE_MISMATCH = 5
BC_STATUS_UNSUPPORTED_DTYPE = 6
BC_STATUS_EXECUTION_FAILED = 7

BC_CMD_ALLOC = 1
BC_DTYPE_INT8 = 1
BC_DTYPE_INT32 = 3
BC_LAYOUT_ROW_MAJOR = 1
BC_USAGE_ACTIVATION = 2
BC_USAGE_OUTPUT = 4
BC_USAGE_DESCRIPTOR = 5

BCUDA_MODEL_MAGIC = 0x444D4342
BCUDA_MODEL_VERSION = 1
BC_OP_GEMM = 1
BC_OP_GRAPH = 2
BC_OP_CONV2D_GEMM = 3
BC_OP_ACTIVATION = 4
BC_OP_POOL = 5
BC_OP_UPSAMPLE = 6
BC_OP_CONCAT = 7
BC_OP_RESHAPE = 8
BC_OP_DETECT_DECODE = 9
BC_OP_NMS = 10
BC_OP_REQUANTIZE = 11
BCUDA_DIAGNOSTICS_MAGIC = 0x47444342
BCUDA_DIAGNOSTICS_VERSION = 2
BCUDA_DIAG_TIMING = 1
BCUDA_DIAG_MEMORY = 2
BCUDA_DIAG_OPERATIONS = 4
BCUDA_DIAG_UTILIZATION = 8
BCUDA_DIAG_ALL = 15


class BcudaHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint16),
        ("command", ctypes.c_uint16),
        ("request_id", ctypes.c_uint32),
        ("header_bytes", ctypes.c_uint32),
        ("payload_bytes", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("checksum", ctypes.c_uint32),
    ]


class BcudaResponseHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint16),
        ("command", ctypes.c_uint16),
        ("request_id", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("header_bytes", ctypes.c_uint32),
        ("payload_bytes", ctypes.c_uint32),
        ("error_code", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class BcudaTensorDesc(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("dtype", ctypes.c_uint32),
        ("rank", ctypes.c_uint32),
        ("dim0", ctypes.c_uint32),
        ("dim1", ctypes.c_uint32),
        ("dim2", ctypes.c_uint32),
        ("dim3", ctypes.c_uint32),
        ("layout", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class BcudaBuffer(ctypes.Structure):
    _fields_ = [
        ("buffer_id", ctypes.c_uint32),
        ("physical_address", ctypes.c_size_t),
        ("virtual_address", ctypes.c_void_p),
        ("size_bytes", ctypes.c_uint64),
        ("dtype", ctypes.c_uint32),
        ("layout", ctypes.c_uint32),
        ("usage", ctypes.c_uint32),
        ("rank", ctypes.c_uint32),
        ("shape", ctypes.c_uint32 * 4),
        ("state", ctypes.c_uint32),
    ]


class BcudaGemmDesc(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("A_buffer", ctypes.c_uint32),
        ("B_buffer", ctypes.c_uint32),
        ("C_buffer", ctypes.c_uint32),
        ("M", ctypes.c_uint32),
        ("N", ctypes.c_uint32),
        ("K", ctypes.c_uint32),
        ("A_dtype", ctypes.c_uint32),
        ("B_dtype", ctypes.c_uint32),
        ("C_dtype", ctypes.c_uint32),
        ("A_layout", ctypes.c_uint32),
        ("B_layout", ctypes.c_uint32),
        ("C_layout", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class BcudaTileLoadDesc(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("buffer_id", ctypes.c_uint32),
        ("matrix_select", ctypes.c_uint32),
        ("base_offset_bytes", ctypes.c_uint64),
        ("row_stride_bytes", ctypes.c_uint32),
        ("tile_rows", ctypes.c_uint32),
        ("tile_cols", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class BcudaTiledGemmConfig(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("tile_size", ctypes.c_uint32),
        ("array_count", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class BcudaAccumulatorConfig(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("tile_entries", ctypes.c_uint32),
        ("clear_on_first_pass", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class BcudaMemoryIoConfig(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("memory_base", ctypes.c_uint64),
        ("memory_bytes", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class BcudaGraphExecDesc(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("model_desc_buffer", ctypes.c_uint32),
        ("input_buffer", ctypes.c_uint32),
        ("output_buffer", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("diagnostics_buffer", ctypes.c_uint32),
    ]


class BcudaDiagnosticsHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("layer_capacity", ctypes.c_uint32),
        ("layer_count", ctypes.c_uint32),
        ("record_bytes", ctypes.c_uint32),
        ("clock_hz", ctypes.c_uint32),
        ("peak_operations_per_cycle", ctypes.c_uint32),
        ("total_elapsed_ns", ctypes.c_uint64),
        ("arena_bytes_used", ctypes.c_uint64),
    ]


class BcudaDiagnosticsRecord(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("layer_index", ctypes.c_uint32),
        ("op_type", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("elapsed_ns", ctypes.c_uint64),
        ("cycles", ctypes.c_uint64),
        ("input_bytes", ctypes.c_uint64),
        ("output_bytes", ctypes.c_uint64),
        ("weight_bytes", ctypes.c_uint64),
        ("operations", ctypes.c_uint64),
        ("throughput_ops_per_second", ctypes.c_uint64),
        ("utilization_ppm", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32),
        ("conv_cycles_total", ctypes.c_uint32),
        ("conv_cycles_zero_a", ctypes.c_uint32),
        ("conv_cycles_load_a", ctypes.c_uint32),
        ("conv_cycles_zero_b", ctypes.c_uint32),
        ("conv_cycles_load_b", ctypes.c_uint32),
        ("conv_cycles_bias", ctypes.c_uint32),
        ("conv_cycles_compute", ctypes.c_uint32),
        ("conv_cycles_accumulate", ctypes.c_uint32),
        ("conv_cycles_write", ctypes.c_uint32),
        ("conv_loader_address_cycles", ctypes.c_uint32),
        ("conv_loader_request_cycles", ctypes.c_uint32),
        ("conv_loader_receive_cycles", ctypes.c_uint32),
        ("conv_cache_hits", ctypes.c_uint32),
        ("conv_cache_misses", ctypes.c_uint32),
        ("conv_fill_cycles", ctypes.c_uint32),
        ("conv_bytes_delivered", ctypes.c_uint32),
    ]


class BcudaModelDesc(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("num_layers", ctypes.c_uint32),
        ("layer_desc_offset", ctypes.c_uint32),
        ("num_buffers", ctypes.c_uint32),
        ("buffer_table_offset", ctypes.c_uint32),
        ("input_buffer", ctypes.c_uint32),
        ("output_buffer", ctypes.c_uint32),
        ("activation_arena_a", ctypes.c_uint32),
        ("activation_arena_b", ctypes.c_uint32),
        ("workspace_buffer", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32),
    ]


class BcudaModelBufferDesc(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("buffer_id", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("layout", ctypes.c_uint32),
        ("usage", ctypes.c_uint32),
        ("size_bytes", ctypes.c_uint32),
        ("rank", ctypes.c_uint32),
        ("dim0", ctypes.c_uint32),
        ("dim1", ctypes.c_uint32),
    ]


class BcudaLayerDesc(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("op_type", ctypes.c_uint32),
        ("input0", ctypes.c_uint32),
        ("input1", ctypes.c_uint32),
        ("input2", ctypes.c_uint32),
        ("output", ctypes.c_uint32),
        ("weight", ctypes.c_uint32),
        ("bias", ctypes.c_uint32),
        ("scale", ctypes.c_uint32),
        ("n", ctypes.c_uint32),
        ("c", ctypes.c_uint32),
        ("h", ctypes.c_uint32),
        ("w", ctypes.c_uint32),
        ("out_c", ctypes.c_uint32),
        ("out_h", ctypes.c_uint32),
        ("out_w", ctypes.c_uint32),
        ("kernel_h", ctypes.c_uint32),
        ("kernel_w", ctypes.c_uint32),
        ("stride_h", ctypes.c_uint32),
        ("stride_w", ctypes.c_uint32),
        ("pad_h", ctypes.c_uint32),
        ("pad_w", ctypes.c_uint32),
        ("groups", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class BcudaAxiDma(ctypes.Structure):
    _fields_ = [
        ("register_base", ctypes.c_size_t),
        ("timeout_cycles", ctypes.c_uint32),
        ("last_control_word", ctypes.c_uint32),
        ("last_source_address", ctypes.c_size_t),
        ("last_transfer_bytes", ctypes.c_uint32),
        ("last_tile_size", ctypes.c_uint32),
        ("last_array_count", ctypes.c_uint32),
        ("last_m_tiles", ctypes.c_uint32),
        ("last_n_tiles", ctypes.c_uint32),
        ("last_k_tiles", ctypes.c_uint32),
        ("last_n_groups", ctypes.c_uint32),
        ("last_schedule_steps", ctypes.c_uint32),
        ("last_accumulator_entries", ctypes.c_uint32),
        ("last_memory_base", ctypes.c_size_t),
        ("last_memory_bytes", ctypes.c_uint32),
        ("last_memory_io_flags", ctypes.c_uint32),
        ("conv_cycles_total", ctypes.c_uint32),
        ("conv_cycles_zero_a", ctypes.c_uint32),
        ("conv_cycles_load_a", ctypes.c_uint32),
        ("conv_cycles_zero_b", ctypes.c_uint32),
        ("conv_cycles_load_b", ctypes.c_uint32),
        ("conv_cycles_bias", ctypes.c_uint32),
        ("conv_cycles_compute", ctypes.c_uint32),
        ("conv_cycles_accumulate", ctypes.c_uint32),
        ("conv_cycles_write", ctypes.c_uint32),
        ("conv_loader_address_cycles", ctypes.c_uint32),
        ("conv_loader_request_cycles", ctypes.c_uint32),
        ("conv_loader_receive_cycles", ctypes.c_uint32),
        ("conv_cache_hits", ctypes.c_uint32),
        ("conv_cache_misses", ctypes.c_uint32),
        ("conv_fill_cycles", ctypes.c_uint32),
        ("conv_bytes_delivered", ctypes.c_uint32),
    ]


class BcudaGraphRuntime(ctypes.Structure):
    _fields_ = [("dma", ctypes.POINTER(BcudaAxiDma))]


class BcudaAllocRequest(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("header", BcudaHeader),
        ("size_bytes", ctypes.c_uint64),
        ("tensor", BcudaTensorDesc),
        ("usage", ctypes.c_uint32),
        ("alignment", ctypes.c_uint32),
    ]


class BcudaAllocResponse(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("header", BcudaResponseHeader),
        ("buffer_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("size_bytes", ctypes.c_uint64),
    ]


class BcudaWriteRequest(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("header", BcudaHeader),
        ("buffer_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("offset_bytes", ctypes.c_uint64),
        ("write_bytes", ctypes.c_uint64),
    ]


class BcudaWriteResponse(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("header", BcudaResponseHeader),
        ("buffer_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("written_bytes", ctypes.c_uint64),
    ]


class BcudaReadRequest(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("header", BcudaHeader),
        ("buffer_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("offset_bytes", ctypes.c_uint64),
        ("read_bytes", ctypes.c_uint64),
    ]


class BcudaReadResponse(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("header", BcudaResponseHeader),
        ("buffer_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("read_bytes", ctypes.c_uint64),
    ]


class BcudaExecuteRequest(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("header", BcudaHeader),
        ("op_type", ctypes.c_uint32),
        ("descriptor_bytes", ctypes.c_uint32),
    ]


class BcudaExecuteResponse(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("header", BcudaResponseHeader),
        ("op_type", ctypes.c_uint32),
        ("execution_status", ctypes.c_uint32),
        ("elapsed_cycles", ctypes.c_uint64),
    ]


class BcudaFreeRequest(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("header", BcudaHeader),
        ("buffer_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class BcudaFreeResponse(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("header", BcudaResponseHeader),
        ("buffer_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


def header(command, request_id=1, payload_bytes=0):
    return BcudaHeader(0x42435544, 1, command, request_id, ctypes.sizeof(BcudaHeader), payload_bytes, 0, 0)


def compile_firmware_library(name: str):
    tmp = Path(tempfile.mkdtemp(prefix=f"bcuda_{name}_"))
    out = tmp / f"{name}.dll"
    cmd = [
        "gcc",
        "-shared",
        "-o",
        str(out),
        "-I",
        str(INCLUDE),
        str(SRC / "protocol.c"),
        str(SRC / "buffer_manager.c"),
        str(SRC / "axi_dma.c"),
        str(SRC / "graph_executor.c"),
        str(SRC / "command_dispatch.c"),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return ctypes.CDLL(str(out))


def tensor_desc(dtype=BC_DTYPE_INT8, rank=1, dim0=4, dim1=1, dim2=1, dim3=1):
    return BcudaTensorDesc(dtype, rank, dim0, dim1, dim2, dim3, BC_LAYOUT_ROW_MAJOR, 0)


def init_buffer_manager(lib, arena_size=4096):
    arena = (ctypes.c_uint8 * arena_size)()
    lib.bcuda_buffer_manager_init.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64]
    lib.bcuda_buffer_manager_init(arena, arena_size)
    return arena


def alloc_buffer(lib, size, desc=None, usage=BC_USAGE_ACTIVATION):
    out_id = ctypes.c_uint32()
    desc = desc or tensor_desc(dim0=size)
    status = lib.bcuda_alloc_buffer(size, desc, usage, ctypes.byref(out_id))
    if status != BC_STATUS_OK:
        raise AssertionError(f"alloc failed: {status}")
    return out_id.value


def write_buffer(lib, buffer_id, data: bytes, offset=0):
    payload = ctypes.create_string_buffer(data)
    status = lib.bcuda_write_buffer(buffer_id, offset, payload, len(data))
    if status != BC_STATUS_OK:
        raise AssertionError(f"write failed: {status}")


def read_buffer(lib, buffer_id, size, offset=0) -> bytes:
    payload = ctypes.create_string_buffer(size)
    status = lib.bcuda_read_buffer(buffer_id, offset, payload, size)
    if status != BC_STATUS_OK:
        raise AssertionError(f"read failed: {status}")
    return payload.raw
