import os
import socket
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda import BoardEndpoint, DEFAULT_PYNQ_IP, Device, ProtocolError
from budgetcuda.dtypes import Command
from budgetcuda.protocol import SocketTransport, build_ok_response
from budgetcuda.serializer import ALLOC_RESPONSE_TAIL_STRUCT, HEADER_STRUCT


class SocketTransportTest(unittest.TestCase):
    def setUp(self):
        self._saved = {name: os.environ.get(name) for name in ("BCUDA_BOARD_IP", "BCUDA_BOARD_PORT", "BCUDA_SOCKET_TIMEOUT")}

    def tearDown(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_endpoint_defaults_and_environment_overrides(self):
        self.assertEqual(BoardEndpoint.from_environment().address, DEFAULT_PYNQ_IP)
        os.environ.update(BCUDA_BOARD_IP="10.1.2.3", BCUDA_BOARD_PORT="9123", BCUDA_SOCKET_TIMEOUT="1.25")
        endpoint = BoardEndpoint.from_environment()
        self.assertEqual((endpoint.address, endpoint.port, endpoint.timeout), ("10.1.2.3", 9123, 1.25))
        self.assertEqual(Device().protocol.transport.endpoint, endpoint)

    def test_client_completes_alloc_over_real_tcp(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def serve_once():
            try:
                connection, _ = listener.accept()
                with connection:
                    header = connection.recv(HEADER_STRUCT.size)
                    _, _, command, request_id, header_bytes, payload_bytes, _, _ = HEADER_STRUCT.unpack(header)
                    remaining = header_bytes + payload_bytes - len(header)
                    while remaining:
                        data = connection.recv(remaining)
                        if not data:
                            return
                        remaining -= len(data)
                    self.assertEqual(command, Command.ALLOC)
                    response = build_ok_response(Command.ALLOC, request_id, ALLOC_RESPONSE_TAIL_STRUCT.pack(7, 0, 64))
                    connection.sendall(response)
            finally:
                listener.close()

        worker = threading.Thread(target=serve_once)
        worker.start()
        device = Device(address="127.0.0.1", port=port)
        tensor = device.alloc((64,), "int8")
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual((tensor.buffer_id, tensor.size_bytes), (7, 64))

    def test_transport_rejects_response_command_mismatch(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def serve_wrong_command():
            try:
                connection, _ = listener.accept()
                with connection:
                    request = connection.recv(64)
                    request_id = int.from_bytes(request[8:12], "little")
                    connection.sendall(build_ok_response(Command.FREE, request_id, b"\0" * 8))
            finally:
                listener.close()

        worker = threading.Thread(target=serve_wrong_command)
        worker.start()
        transport = SocketTransport("127.0.0.1", port, timeout=1)
        request = bytearray(12)
        request[6:8] = int(Command.ALLOC).to_bytes(2, "little")
        with self.assertRaisesRegex(ProtocolError, "Response command mismatch"):
            transport.transact(bytes(request))
        worker.join(timeout=1)

    def test_transport_reuses_one_tcp_connection(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        accepts = []

        def serve_two():
            try:
                connection, _ = listener.accept()
                accepts.append(1)
                with connection:
                    for buffer_id in (7, 8):
                        header = connection.recv(HEADER_STRUCT.size)
                        _, _, command, request_id, header_bytes, payload_bytes, _, _ = HEADER_STRUCT.unpack(header)
                        remaining = header_bytes + payload_bytes - len(header)
                        while remaining:
                            data = connection.recv(remaining)
                            if not data:
                                return
                            remaining -= len(data)
                        response = build_ok_response(
                            Command.ALLOC, request_id,
                            ALLOC_RESPONSE_TAIL_STRUCT.pack(buffer_id, 0, 64))
                        connection.sendall(response)
            finally:
                listener.close()

        worker = threading.Thread(target=serve_two)
        worker.start()
        device = Device(address="127.0.0.1", port=port)
        first = device.alloc((64,), "int8")
        second = device.alloc((64,), "int8")
        device.close()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(accepts, [1])
        self.assertEqual((first.buffer_id, second.buffer_id), (7, 8))


if __name__ == "__main__":
    unittest.main()
