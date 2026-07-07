import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda.dtypes import OpType
from budgetcuda.mock import MockBackend
from budgetcuda.protocol import ProtocolClient
from budgetcuda.serializer import pack_gemm_desc


class ProtocolTest(unittest.TestCase):
    def test_protocol_client_alloc_write_read_free_with_mock_backend(self):
        client = ProtocolClient(transport=MockBackend())
        desc = client.tensor_desc((4,), "int8")

        alloc = client.alloc(4, desc, "activation")
        buffer_id = alloc.fields["buffer_id"]

        write = client.write(buffer_id, bytes([1, 2, 3, 4]))
        self.assertEqual(write.fields["written_bytes"], 4)

        raw = client.read(buffer_id, 0, 4)
        self.assertEqual(raw, bytes([1, 2, 3, 4]))

        freed = client.free(buffer_id)
        self.assertEqual(freed.fields["buffer_id"], buffer_id)

    def test_protocol_client_execute_gemm_descriptor_with_mock_backend(self):
        client = ProtocolClient(transport=MockBackend())
        A = client.alloc(4, client.tensor_desc((2, 2), "int8"), "activation").fields["buffer_id"]
        B = client.alloc(4, client.tensor_desc((2, 2), "int8"), "weight").fields["buffer_id"]
        C = client.alloc(16, client.tensor_desc((2, 2), "int32"), "output").fields["buffer_id"]

        client.write(A, bytes([1, 2, 3, 4]))
        client.write(B, bytes([5, 6, 7, 8]))

        response = client.execute(OpType.GEMM, pack_gemm_desc(A, B, C, 2, 2, 2))
        self.assertEqual(response.fields["op_type"], OpType.GEMM)

        raw = client.read(C, 0, 16)
        self.assertEqual(
            raw,
            (19).to_bytes(4, "little", signed=True)
            + (22).to_bytes(4, "little", signed=True)
            + (43).to_bytes(4, "little", signed=True)
            + (50).to_bytes(4, "little", signed=True),
        )

    def test_protocol_client_raises_on_invalid_handle_in_mock_backend(self):
        client = ProtocolClient(transport=MockBackend())
        with self.assertRaises(KeyError):
            client.read(999, 0, 1)

    def test_protocol_client_requires_address_or_transport(self):
        with self.assertRaises(ValueError):
            ProtocolClient()


if __name__ == "__main__":
    unittest.main()
