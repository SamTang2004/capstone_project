import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda.dtypes import OpType
from budgetcuda.graph import GemmOp, GraphExec
from budgetcuda.serializer import unpack_gemm_desc


class GraphTest(unittest.TestCase):
    def test_gemm_op_builds_gemm_descriptor(self):
        op = GemmOp(A_buffer=1, B_buffer=2, C_buffer=3, M=4, N=5, K=6)
        self.assertEqual(op.op_type, OpType.GEMM)
        desc = unpack_gemm_desc(op.to_descriptor())
        self.assertEqual(desc["A_buffer"], 1)
        self.assertEqual(desc["M"], 4)
        self.assertEqual(desc["N"], 5)
        self.assertEqual(desc["K"], 6)

    def test_graph_exec_builds_graph_descriptor(self):
        op = GraphExec(model_desc_buffer=10, input_buffer=20, output_buffer=30, flags=7)
        self.assertEqual(op.op_type, OpType.GRAPH)
        self.assertEqual(
            op.to_descriptor(),
            (10).to_bytes(4, "little")
            + (20).to_bytes(4, "little")
            + (30).to_bytes(4, "little")
            + (7).to_bytes(4, "little")
            + (0).to_bytes(4, "little"),
        )


if __name__ == "__main__":
    unittest.main()
