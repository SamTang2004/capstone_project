from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable

from .dtypes import (
    BCUDA_MAGIC,
    BCUDA_VERSION,
    Command,
    Layout,
    OpType,
    normalize_dtype,
    normalize_layout,
)


HEADER_STRUCT = struct.Struct("<IHHIIIII")
RESPONSE_HEADER_STRUCT = struct.Struct("<IHHIIIIII")
TENSOR_DESC_STRUCT = struct.Struct("<IIIIIIII")
ALLOC_REQUEST_TAIL_STRUCT = struct.Struct("<Q")
ALLOC_RESPONSE_TAIL_STRUCT = struct.Struct("<IIQ")
WRITE_REQUEST_TAIL_STRUCT = struct.Struct("<IIQQ")
WRITE_RESPONSE_TAIL_STRUCT = struct.Struct("<IIQ")
READ_REQUEST_TAIL_STRUCT = struct.Struct("<IIQQ")
READ_RESPONSE_TAIL_STRUCT = struct.Struct("<IIQ")
EXECUTE_REQUEST_TAIL_STRUCT = struct.Struct("<II")
EXECUTE_RESPONSE_TAIL_STRUCT = struct.Struct("<IIQ")
FREE_REQUEST_TAIL_STRUCT = struct.Struct("<II")
FREE_RESPONSE_TAIL_STRUCT = struct.Struct("<II")
GEMM_DESC_STRUCT = struct.Struct("<IIIIIIIIIIIIII")
GRAPH_EXEC_DESC_STRUCT = struct.Struct("<IIIII")


@dataclass(frozen=True)
class ResponseHeader:
    magic: int
    version: int
    command: int
    request_id: int
    status: int
    header_bytes: int
    payload_bytes: int
    error_code: int
    reserved: int


def _shape4(shape: Iterable[int]) -> tuple[int, int, int, int]:
    dims = tuple(int(dim) for dim in shape)
    if len(dims) > 4:
        raise ValueError("Budget-CUDA tensor descriptors support rank <= 4")
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"Tensor dimensions must be positive: {dims}")
    return dims + (1,) * (4 - len(dims))


def pack_header(
    command: int | Command,
    request_id: int,
    header_bytes: int,
    payload_bytes: int = 0,
    flags: int = 0,
    checksum: int = 0,
    version: int = BCUDA_VERSION,
) -> bytes:
    return HEADER_STRUCT.pack(
        BCUDA_MAGIC,
        version,
        int(command),
        request_id,
        header_bytes,
        payload_bytes,
        flags,
        checksum,
    )


def pack_response_header(
    command: int | Command,
    request_id: int,
    status: int,
    header_bytes: int,
    payload_bytes: int = 0,
    error_code: int = 0,
    reserved: int = 0,
    version: int = BCUDA_VERSION,
) -> bytes:
    return RESPONSE_HEADER_STRUCT.pack(
        BCUDA_MAGIC,
        version,
        int(command),
        request_id,
        status,
        header_bytes,
        payload_bytes,
        error_code,
        reserved,
    )


def unpack_response_header(data: bytes) -> ResponseHeader:
    if len(data) < RESPONSE_HEADER_STRUCT.size:
        raise ValueError("Not enough bytes for BcudaResponseHeader")
    return ResponseHeader(*RESPONSE_HEADER_STRUCT.unpack_from(data))


def pack_tensor_desc(
    shape: Iterable[int],
    dtype: str | int,
    layout: str | int = Layout.ROW_MAJOR,
) -> bytes:
    shape = tuple(int(dim) for dim in shape)
    dims = _shape4(shape)
    return TENSOR_DESC_STRUCT.pack(
        int(normalize_dtype(dtype)),
        len(shape),
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        int(normalize_layout(layout)),
        0,
    )


def unpack_tensor_desc(data: bytes) -> dict:
    if len(data) < TENSOR_DESC_STRUCT.size:
        raise ValueError("Not enough bytes for BcudaTensorDesc")
    dtype, rank, dim0, dim1, dim2, dim3, layout, reserved = TENSOR_DESC_STRUCT.unpack_from(data)
    return {
        "dtype": dtype,
        "rank": rank,
        "shape": (dim0, dim1, dim2, dim3)[:rank],
        "layout": layout,
        "reserved": reserved,
    }


def pack_alloc_request(
    request_id: int,
    size_bytes: int,
    shape: Iterable[int],
    dtype: str | int,
    usage: int,
    alignment: int = 64,
    layout: str | int = Layout.ROW_MAJOR,
) -> bytes:
    header_bytes = HEADER_STRUCT.size + ALLOC_REQUEST_TAIL_STRUCT.size + TENSOR_DESC_STRUCT.size + 8
    return b"".join(
        [
            pack_header(Command.ALLOC, request_id, header_bytes),
            ALLOC_REQUEST_TAIL_STRUCT.pack(size_bytes),
            pack_tensor_desc(shape, dtype, layout),
            struct.pack("<II", int(usage), alignment),
        ]
    )


def pack_gemm_desc(
    A_buffer: int,
    B_buffer: int,
    C_buffer: int,
    M: int,
    N: int,
    K: int,
    A_dtype: str | int = "int8",
    B_dtype: str | int = "int8",
    C_dtype: str | int = "int32",
    A_layout: str | int = "row_major",
    B_layout: str | int = "row_major",
    C_layout: str | int = "row_major",
    flags: int = 0,
    reserved: int = 0,
) -> bytes:
    return GEMM_DESC_STRUCT.pack(
        A_buffer,
        B_buffer,
        C_buffer,
        M,
        N,
        K,
        int(normalize_dtype(A_dtype)),
        int(normalize_dtype(B_dtype)),
        int(normalize_dtype(C_dtype)),
        int(normalize_layout(A_layout)),
        int(normalize_layout(B_layout)),
        int(normalize_layout(C_layout)),
        flags,
        reserved,
    )


def unpack_gemm_desc(data: bytes) -> dict:
    if len(data) < GEMM_DESC_STRUCT.size:
        raise ValueError("Not enough bytes for BcudaGemmDesc")
    fields = GEMM_DESC_STRUCT.unpack_from(data)
    keys = (
        "A_buffer",
        "B_buffer",
        "C_buffer",
        "M",
        "N",
        "K",
        "A_dtype",
        "B_dtype",
        "C_dtype",
        "A_layout",
        "B_layout",
        "C_layout",
        "flags",
        "reserved",
    )
    return dict(zip(keys, fields))


def pack_graph_exec_desc(
    model_desc_buffer: int,
    input_buffer: int,
    output_buffer: int,
    flags: int = 0,
    reserved: int = 0,
) -> bytes:
    return GRAPH_EXEC_DESC_STRUCT.pack(
        model_desc_buffer,
        input_buffer,
        output_buffer,
        flags,
        reserved,
    )


def pack_execute_request(request_id: int, op_type: int | OpType, descriptor: bytes) -> bytes:
    header_bytes = HEADER_STRUCT.size + EXECUTE_REQUEST_TAIL_STRUCT.size
    return b"".join(
        [
            pack_header(Command.EXECUTE, request_id, header_bytes, payload_bytes=len(descriptor)),
            EXECUTE_REQUEST_TAIL_STRUCT.pack(int(op_type), len(descriptor)),
            descriptor,
        ]
    )
