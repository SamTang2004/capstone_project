import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda.dtypes import BCUDA_MAGIC, BCUDA_VERSION, Command, DType, Layout, OpType
from budgetcuda.serializer import (
    GEMM_DESC_STRUCT,
    HEADER_STRUCT,
    RESPONSE_HEADER_STRUCT,
    TENSOR_DESC_STRUCT,
    pack_execute_request,
    pack_gemm_desc,
    pack_graph_exec_desc,
    pack_header,
    pack_response_header,
    pack_tensor_desc,
    unpack_gemm_desc,
    unpack_response_header,
    unpack_tensor_desc,
)


class SerializerTest(unittest.TestCase):
    def test_pack_header_matches_protocol_layout(self):
        packed = pack_header(Command.ALLOC, request_id=7, header_bytes=76, payload_bytes=12)
        self.assertEqual(len(packed), HEADER_STRUCT.size)
        self.assertEqual(
            HEADER_STRUCT.unpack(packed),
            (BCUDA_MAGIC, BCUDA_VERSION, Command.ALLOC, 7, 76, 12, 0, 0),
        )

    def test_pack_response_header_round_trips(self):
        packed = pack_response_header(Command.READ, 9, 0, RESPONSE_HEADER_STRUCT.size, payload_bytes=4)
        header = unpack_response_header(packed)
        self.assertEqual(header.magic, BCUDA_MAGIC)
        self.assertEqual(header.command, Command.READ)
        self.assertEqual(header.request_id, 9)
        self.assertEqual(header.payload_bytes, 4)

    def test_unpack_response_header_rejects_short_data(self):
        with self.assertRaises(ValueError):
            unpack_response_header(b"short")

    def test_pack_tensor_desc_round_trips_shape_dtype_layout(self):
        packed = pack_tensor_desc((3, 64, 64), "int8", "nchw")
        self.assertEqual(len(packed), TENSOR_DESC_STRUCT.size)
        desc = unpack_tensor_desc(packed)
        self.assertEqual(desc["dtype"], DType.INT8)
        self.assertEqual(desc["rank"], 3)
        self.assertEqual(desc["shape"], (3, 64, 64))
        self.assertEqual(desc["layout"], Layout.NCHW)

    def test_pack_tensor_desc_rejects_rank_greater_than_four(self):
        with self.assertRaises(ValueError):
            pack_tensor_desc((1, 2, 3, 4, 5), "int8")

    def test_pack_gemm_desc_round_trips_all_fields(self):
        packed = pack_gemm_desc(1, 2, 3, 16, 32, 64)
        self.assertEqual(len(packed), GEMM_DESC_STRUCT.size)
        desc = unpack_gemm_desc(packed)
        self.assertEqual(desc["A_buffer"], 1)
        self.assertEqual(desc["B_buffer"], 2)
        self.assertEqual(desc["C_buffer"], 3)
        self.assertEqual(desc["M"], 16)
        self.assertEqual(desc["N"], 32)
        self.assertEqual(desc["K"], 64)
        self.assertEqual(desc["A_dtype"], DType.INT8)
        self.assertEqual(desc["C_dtype"], DType.INT32)

    def test_unpack_gemm_desc_rejects_short_data(self):
        with self.assertRaises(ValueError):
            unpack_gemm_desc(b"short")

    def test_pack_graph_exec_desc_uses_five_u32_fields(self):
        packed = pack_graph_exec_desc(11, 12, 13, flags=3)
        self.assertEqual(len(packed), 20)
        self.assertEqual(
            packed,
            (11).to_bytes(4, "little")
            + (12).to_bytes(4, "little")
            + (13).to_bytes(4, "little")
            + (3).to_bytes(4, "little")
            + (0).to_bytes(4, "little"),
        )

    def test_pack_execute_request_appends_descriptor_payload(self):
        desc = pack_gemm_desc(1, 2, 3, 16, 16, 16)
        request = pack_execute_request(5, OpType.GEMM, desc)
        magic, _, command, request_id, header_bytes, payload_bytes, _, _ = HEADER_STRUCT.unpack_from(request)
        self.assertEqual(magic, BCUDA_MAGIC)
        self.assertEqual(command, Command.EXECUTE)
        self.assertEqual(request_id, 5)
        self.assertEqual(header_bytes, HEADER_STRUCT.size + 8)
        self.assertEqual(payload_bytes, len(desc))
        self.assertEqual(request[-len(desc) :], desc)


if __name__ == "__main__":
    unittest.main()
