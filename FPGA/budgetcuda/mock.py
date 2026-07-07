from __future__ import annotations

import struct

from .dtypes import Command, DType, OpType, Status
from .protocol import build_ok_response
from .serializer import (
    EXECUTE_REQUEST_TAIL_STRUCT,
    EXECUTE_RESPONSE_TAIL_STRUCT,
    FREE_REQUEST_TAIL_STRUCT,
    FREE_RESPONSE_TAIL_STRUCT,
    HEADER_STRUCT,
    READ_REQUEST_TAIL_STRUCT,
    READ_RESPONSE_TAIL_STRUCT,
    RESPONSE_HEADER_STRUCT,
    TENSOR_DESC_STRUCT,
    WRITE_REQUEST_TAIL_STRUCT,
    WRITE_RESPONSE_TAIL_STRUCT,
    unpack_gemm_desc,
)


class MockBackend:
    def __init__(self):
        self._next_buffer_id = 1
        self.buffers: dict[int, bytearray] = {}

    def transact(self, request: bytes) -> bytes:
        header = HEADER_STRUCT.unpack_from(request)
        _, _, command, request_id, header_bytes, payload_bytes, _, _ = header
        payload = request[header_bytes : header_bytes + payload_bytes]

        if command == Command.ALLOC:
            return self._handle_alloc(request, request_id)
        if command == Command.WRITE:
            return self._handle_write(request, request_id, payload)
        if command == Command.READ:
            return self._handle_read(request, request_id)
        if command == Command.EXECUTE:
            return self._handle_execute(request, request_id, payload)
        if command == Command.FREE:
            return self._handle_free(request, request_id)
        return self._error_response(Command(command), request_id, Status.INVALID_COMMAND)

    def _handle_alloc(self, request: bytes, request_id: int) -> bytes:
        offset = HEADER_STRUCT.size
        size_bytes = int.from_bytes(request[offset : offset + 8], "little")
        buffer_id = self._next_buffer_id
        self._next_buffer_id += 1
        self.buffers[buffer_id] = bytearray(size_bytes)
        tail = struct.pack("<IIQ", buffer_id, 0, size_bytes)
        return build_ok_response(Command.ALLOC, request_id, tail)

    def _handle_write(self, request: bytes, request_id: int, payload: bytes) -> bytes:
        offset = HEADER_STRUCT.size
        buffer_id, _, offset_bytes, write_bytes = WRITE_REQUEST_TAIL_STRUCT.unpack_from(request, offset)
        buf = self._get_buffer(buffer_id)
        end = offset_bytes + write_bytes
        if end > len(buf):
            return self._error_response(Command.WRITE, request_id, Status.GENERAL_ERROR)
        buf[offset_bytes:end] = payload[:write_bytes]
        tail = WRITE_RESPONSE_TAIL_STRUCT.pack(buffer_id, 0, write_bytes)
        return build_ok_response(Command.WRITE, request_id, tail)

    def _handle_read(self, request: bytes, request_id: int) -> bytes:
        offset = HEADER_STRUCT.size
        buffer_id, _, offset_bytes, read_bytes = READ_REQUEST_TAIL_STRUCT.unpack_from(request, offset)
        buf = self._get_buffer(buffer_id)
        end = offset_bytes + read_bytes
        if end > len(buf):
            return self._error_response(Command.READ, request_id, Status.GENERAL_ERROR)
        payload = bytes(buf[offset_bytes:end])
        tail = READ_RESPONSE_TAIL_STRUCT.pack(buffer_id, 0, read_bytes)
        return build_ok_response(Command.READ, request_id, tail, payload)

    def _handle_execute(self, request: bytes, request_id: int, payload: bytes) -> bytes:
        offset = HEADER_STRUCT.size
        op_type, descriptor_bytes = EXECUTE_REQUEST_TAIL_STRUCT.unpack_from(request, offset)
        descriptor = payload[:descriptor_bytes]
        if op_type == OpType.GEMM:
            self._execute_gemm(descriptor)
        else:
            return self._error_response(Command.EXECUTE, request_id, Status.INVALID_COMMAND)
        tail = EXECUTE_RESPONSE_TAIL_STRUCT.pack(op_type, Status.OK, 0)
        return build_ok_response(Command.EXECUTE, request_id, tail)

    def _handle_free(self, request: bytes, request_id: int) -> bytes:
        offset = HEADER_STRUCT.size
        buffer_id, _ = FREE_REQUEST_TAIL_STRUCT.unpack_from(request, offset)
        self._get_buffer(buffer_id)
        del self.buffers[buffer_id]
        tail = FREE_RESPONSE_TAIL_STRUCT.pack(buffer_id, 0)
        return build_ok_response(Command.FREE, request_id, tail)

    def _execute_gemm(self, descriptor: bytes) -> None:
        desc = unpack_gemm_desc(descriptor)
        if desc["A_dtype"] != DType.INT8 or desc["B_dtype"] != DType.INT8 or desc["C_dtype"] != DType.INT32:
            raise ValueError("Mock GEMM only supports INT8 x INT8 -> INT32")
        M, N, K = desc["M"], desc["N"], desc["K"]
        A = self._read_int8_matrix(desc["A_buffer"], M, K)
        B = self._read_int8_matrix(desc["B_buffer"], K, N)
        out = bytearray()
        for m in range(M):
            for n in range(N):
                acc = 0
                for k in range(K):
                    acc += A[m][k] * B[k][n]
                out += int(acc).to_bytes(4, "little", signed=True)
        self._get_buffer(desc["C_buffer"])[: len(out)] = out

    def _read_int8_matrix(self, buffer_id: int, rows: int, cols: int) -> list[list[int]]:
        raw = self._get_buffer(buffer_id)
        values = [byte if byte < 128 else byte - 256 for byte in raw[: rows * cols]]
        return [values[row * cols : (row + 1) * cols] for row in range(rows)]

    def _get_buffer(self, buffer_id: int) -> bytearray:
        try:
            return self.buffers[int(buffer_id)]
        except KeyError as exc:
            raise KeyError(f"Invalid mock buffer id: {buffer_id}") from exc

    @staticmethod
    def _error_response(command: Command, request_id: int, status: Status) -> bytes:
        return b"".join(
            [
                struct.pack(
                    "<IHHIIIIII",
                    0x42435544,
                    1,
                    int(command),
                    request_id,
                    int(status),
                    RESPONSE_HEADER_STRUCT.size,
                    0,
                    int(status),
                    0,
                )
            ]
        )
