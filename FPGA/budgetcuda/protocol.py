from __future__ import annotations

import os
import socket
from dataclasses import dataclass

from .dtypes import BCUDA_MAGIC, BCUDA_VERSION, Command, OpType, Status, normalize_usage
from .serializer import (
    EXECUTE_RESPONSE_TAIL_STRUCT,
    FREE_RESPONSE_TAIL_STRUCT,
    READ_RESPONSE_TAIL_STRUCT,
    RESPONSE_HEADER_STRUCT,
    WRITE_RESPONSE_TAIL_STRUCT,
    pack_alloc_request,
    pack_execute_request,
    pack_header,
    pack_response_header,
    pack_tensor_desc,
    unpack_response_header,
)


class ProtocolError(RuntimeError):
    pass


DEFAULT_PYNQ_IP = "192.168.2.99"
DEFAULT_PROTOCOL_PORT = 9000
DEFAULT_SOCKET_TIMEOUT = 5.0
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class BoardEndpoint:
    """Network location of the BudgetCUDA runtime running on the board PS."""

    address: str = DEFAULT_PYNQ_IP
    port: int = DEFAULT_PROTOCOL_PORT
    timeout: float = DEFAULT_SOCKET_TIMEOUT

    @classmethod
    def from_environment(
        cls,
        address: str | None = None,
        port: int | None = None,
        timeout: float | None = None,
    ) -> "BoardEndpoint":
        address = address or os.environ.get("BCUDA_BOARD_IP") or os.environ.get("BUDGETCUDA_BOARD_IP") or DEFAULT_PYNQ_IP
        port = port if port is not None else os.environ.get("BCUDA_BOARD_PORT", os.environ.get("BUDGETCUDA_BOARD_PORT", DEFAULT_PROTOCOL_PORT))
        timeout = timeout if timeout is not None else os.environ.get("BCUDA_SOCKET_TIMEOUT", DEFAULT_SOCKET_TIMEOUT)
        try:
            port = int(port)
            timeout = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("BudgetCUDA board port and socket timeout must be numeric") from exc
        if not address or not 1 <= port <= 65535 or timeout <= 0:
            raise ValueError("Invalid BudgetCUDA board endpoint")
        return cls(str(address), port, timeout)


@dataclass
class ProtocolResponse:
    status: int
    payload: bytes = b""
    fields: dict | None = None


class SocketTransport:
    """One-request TCP transport for the board-resident BudgetCUDA runtime."""

    def __init__(self, address: str | None = None, port: int | None = None, timeout: float | None = None):
        self.endpoint = BoardEndpoint.from_environment(address, port, timeout)
        self.address = self.endpoint.address
        self.port = self.endpoint.port
        self.timeout = self.endpoint.timeout

    def transact(self, request: bytes) -> bytes:
        if len(request) < 8:
            raise ProtocolError("Request is too short to contain a BudgetCUDA command")
        expected_command = int.from_bytes(request[6:8], "little")
        try:
            with socket.create_connection((self.address, self.port), timeout=self.timeout) as sock:
                sock.settimeout(self.timeout)
                sock.sendall(request)
                header = self._recv_exact(sock, RESPONSE_HEADER_STRUCT.size)
                response_header = unpack_response_header(header)
                self._validate_envelope(response_header, expected_command)
                trailing_bytes = response_header.header_bytes - RESPONSE_HEADER_STRUCT.size + response_header.payload_bytes
                tail_and_payload = self._recv_exact(sock, trailing_bytes)
                return header + tail_and_payload
        except (OSError, socket.timeout) as exc:
            raise ProtocolError(f"Cannot reach BudgetCUDA board at {self.address}:{self.port}: {exc}") from exc

    @staticmethod
    def _validate_envelope(response_header, expected_command: int) -> None:
        if response_header.magic != BCUDA_MAGIC or response_header.version != BCUDA_VERSION:
            raise ProtocolError("BudgetCUDA board returned an incompatible protocol header")
        if response_header.command != expected_command:
            raise ProtocolError(f"Response command mismatch: {response_header.command} != {expected_command}")
        if response_header.header_bytes < RESPONSE_HEADER_STRUCT.size:
            raise ProtocolError("BudgetCUDA board returned an invalid response header size")
        total_bytes = response_header.header_bytes + response_header.payload_bytes
        if total_bytes > MAX_RESPONSE_BYTES:
            raise ProtocolError(f"BudgetCUDA response exceeds {MAX_RESPONSE_BYTES} byte safety limit")

    @staticmethod
    def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
        chunks = []
        remaining = nbytes
        while remaining:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ProtocolError("Connection closed while receiving response")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


class ProtocolClient:
    def __init__(self, address: str | None = None, port: int | None = None, timeout: float | None = None, transport=None):
        self._request_id = 0
        if transport is not None:
            self.transport = transport
        else:
            self.transport = SocketTransport(address, port, timeout)

    def next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def alloc(self, size_bytes, tensor_desc, usage, alignment=64):
        req_id = self.next_request_id()
        request = pack_alloc_request(
            req_id,
            int(size_bytes),
            tensor_desc["shape"],
            tensor_desc["dtype"],
            int(normalize_usage(usage)),
            alignment,
            tensor_desc.get("layout", "row_major"),
        )
        response = self.transport.transact(request)
        return self._decode_alloc_response(response, req_id)

    def write(self, buffer_id, data: bytes, offset_bytes=0):
        req_id = self.next_request_id()
        header_bytes = 28 + 24
        request = b"".join(
            [
                pack_header(Command.WRITE, req_id, header_bytes, payload_bytes=len(data)),
                int(buffer_id).to_bytes(4, "little"),
                (0).to_bytes(4, "little"),
                int(offset_bytes).to_bytes(8, "little"),
                len(data).to_bytes(8, "little"),
                data,
            ]
        )
        response = self.transport.transact(request)
        return self._decode_write_response(response, req_id)

    def read(self, buffer_id, offset_bytes, read_bytes) -> bytes:
        req_id = self.next_request_id()
        header_bytes = 28 + 24
        request = b"".join(
            [
                pack_header(Command.READ, req_id, header_bytes),
                int(buffer_id).to_bytes(4, "little"),
                (0).to_bytes(4, "little"),
                int(offset_bytes).to_bytes(8, "little"),
                int(read_bytes).to_bytes(8, "little"),
            ]
        )
        response = self.transport.transact(request)
        decoded = self._decode_read_response(response, req_id)
        return decoded.payload

    def execute(self, op_type, descriptor: bytes):
        req_id = self.next_request_id()
        request = pack_execute_request(req_id, op_type, descriptor)
        response = self.transport.transact(request)
        return self._decode_execute_response(response, req_id)

    def free(self, buffer_id):
        req_id = self.next_request_id()
        header_bytes = 28 + 8
        request = b"".join(
            [
                pack_header(Command.FREE, req_id, header_bytes),
                int(buffer_id).to_bytes(4, "little"),
                (0).to_bytes(4, "little"),
            ]
        )
        response = self.transport.transact(request)
        return self._decode_free_response(response, req_id)

    @staticmethod
    def tensor_desc(shape, dtype, layout="row_major") -> dict:
        # Validate by packing once; the dict is friendlier for Device code.
        pack_tensor_desc(shape, dtype, layout)
        return {"shape": tuple(shape), "dtype": dtype, "layout": layout}

    def _decode_header(self, response: bytes, request_id: int):
        header = unpack_response_header(response)
        if header.magic != BCUDA_MAGIC or header.version != BCUDA_VERSION:
            raise ProtocolError("Response has an incompatible BudgetCUDA protocol header")
        if header.header_bytes < RESPONSE_HEADER_STRUCT.size or header.header_bytes + header.payload_bytes != len(response):
            raise ProtocolError("Response length does not match its BudgetCUDA protocol header")
        if header.request_id != request_id:
            raise ProtocolError(f"Response request_id mismatch: {header.request_id} != {request_id}")
        if header.status != Status.OK:
            raise ProtocolError(f"Device returned status {header.status} (error {header.error_code})")
        return header

    def _decode_alloc_response(self, response: bytes, request_id: int):
        header = self._decode_header(response, request_id)
        offset = RESPONSE_HEADER_STRUCT.size
        buffer_id = int.from_bytes(response[offset : offset + 4], "little")
        size_bytes = int.from_bytes(response[offset + 8 : offset + 16], "little")
        return ProtocolResponse(header.status, fields={"buffer_id": buffer_id, "size_bytes": size_bytes})

    def _decode_write_response(self, response: bytes, request_id: int):
        header = self._decode_header(response, request_id)
        offset = RESPONSE_HEADER_STRUCT.size
        buffer_id, _, written_bytes = WRITE_RESPONSE_TAIL_STRUCT.unpack_from(response, offset)
        return ProtocolResponse(header.status, fields={"buffer_id": buffer_id, "written_bytes": written_bytes})

    def _decode_read_response(self, response: bytes, request_id: int):
        header = self._decode_header(response, request_id)
        offset = RESPONSE_HEADER_STRUCT.size
        buffer_id, _, read_bytes = READ_RESPONSE_TAIL_STRUCT.unpack_from(response, offset)
        payload_offset = offset + READ_RESPONSE_TAIL_STRUCT.size
        return ProtocolResponse(
            header.status,
            payload=response[payload_offset : payload_offset + read_bytes],
            fields={"buffer_id": buffer_id, "read_bytes": read_bytes},
        )

    def _decode_execute_response(self, response: bytes, request_id: int):
        header = self._decode_header(response, request_id)
        offset = RESPONSE_HEADER_STRUCT.size
        op_type, execution_status, elapsed_cycles = EXECUTE_RESPONSE_TAIL_STRUCT.unpack_from(response, offset)
        return ProtocolResponse(
            header.status,
            fields={
                "op_type": op_type,
                "execution_status": execution_status,
                "elapsed_cycles": elapsed_cycles,
            },
        )

    def _decode_free_response(self, response: bytes, request_id: int):
        header = self._decode_header(response, request_id)
        offset = RESPONSE_HEADER_STRUCT.size
        buffer_id, _ = FREE_RESPONSE_TAIL_STRUCT.unpack_from(response, offset)
        return ProtocolResponse(header.status, fields={"buffer_id": buffer_id})


def build_ok_response(command: Command, request_id: int, tail: bytes, payload: bytes = b"") -> bytes:
    return b"".join(
        [
            pack_response_header(
                command,
                request_id,
                Status.OK,
                RESPONSE_HEADER_STRUCT.size + len(tail),
                payload_bytes=len(payload),
            ),
            tail,
            payload,
        ]
    )
