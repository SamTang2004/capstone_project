from __future__ import annotations

import socket
from dataclasses import dataclass

from .dtypes import Command, OpType, Status, normalize_usage
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


@dataclass
class ProtocolResponse:
    status: int
    payload: bytes = b""
    fields: dict | None = None


class SocketTransport:
    def __init__(self, address: str, port: int, timeout: float = 5.0):
        self.address = address
        self.port = int(port)
        self.timeout = timeout

    def transact(self, request: bytes) -> bytes:
        with socket.create_connection((self.address, self.port), timeout=self.timeout) as sock:
            sock.sendall(request)
            header = self._recv_exact(sock, RESPONSE_HEADER_STRUCT.size)
            response_header = unpack_response_header(header)
            tail_and_payload = self._recv_exact(
                sock,
                response_header.header_bytes - RESPONSE_HEADER_STRUCT.size
                + response_header.payload_bytes,
            )
            return header + tail_and_payload

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
    def __init__(self, address: str | None = None, port: int = 9000, transport=None):
        self._request_id = 0
        if transport is not None:
            self.transport = transport
        elif address is not None:
            self.transport = SocketTransport(address, port)
        else:
            raise ValueError("ProtocolClient requires either address or transport")

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
        if header.request_id != request_id:
            raise ProtocolError(f"Response request_id mismatch: {header.request_id} != {request_id}")
        if header.status != Status.OK:
            raise ProtocolError(f"Device returned status {header.status}")
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
