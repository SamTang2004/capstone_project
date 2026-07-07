import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda import Device
from budgetcuda.dtypes import DType, Layout


class DeviceTest(unittest.TestCase):
    def test_device_alloc_returns_tensor_metadata(self):
        dev = Device.mock()
        tensor = dev.alloc((2, 3), "int8", usage="activation")
        self.assertGreater(tensor.buffer_id, 0)
        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.dtype, DType.INT8)
        self.assertEqual(tensor.layout, Layout.ROW_MAJOR)
        self.assertEqual(tensor.nbytes, 6)
        self.assertEqual(tensor.itemsize, 1)
        self.assertEqual(tensor.ndim, 2)

    def test_device_write_and_read_round_trip_numpy_array(self):
        dev = Device.mock()
        tensor = dev.alloc((2, 2), "int8")
        values = [[1, -2], [3, -4]]
        dev.write(tensor, values)
        out = dev.read(tensor)
        self.assertEqual(list(out.reshape(-1)) if hasattr(out, "reshape") else out, [1, -2, 3, -4])

    def test_device_execute_gemm_matches_numpy_reference(self):
        dev = Device.mock()
        A_host = [[1, 2, 3], [4, 5, 6]]
        B_host = [[7, 8], [9, 10], [11, 12]]

        A = dev.alloc((2, 3), "int8")
        B = dev.alloc((3, 2), "int8", usage="weight")
        C = dev.alloc((2, 2), "int32", usage="output")

        dev.write(A, A_host)
        dev.write(B, B_host)
        dev.execute_gemm(A, B, C, M=2, N=2, K=3)
        out = dev.read(C)

        self.assertEqual(list(out.reshape(-1)) if hasattr(out, "reshape") else out, [58, 64, 139, 154])

    def test_device_free_releases_buffer(self):
        dev = Device.mock()
        tensor = dev.alloc((4,), "int8")
        response = dev.free(tensor)
        self.assertEqual(response.fields["buffer_id"], tensor.buffer_id)


if __name__ == "__main__":
    unittest.main()
