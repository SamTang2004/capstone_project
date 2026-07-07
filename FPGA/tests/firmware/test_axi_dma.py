import ctypes
import unittest

from common import (
    BC_DTYPE_INT8,
    BC_DTYPE_INT32,
    BC_STATUS_EXECUTION_FAILED,
    BC_STATUS_OK,
    BC_STATUS_UNSUPPORTED_DTYPE,
    BcudaAxiDma,
    BcudaBuffer,
    BcudaGemmDesc,
    compile_firmware_library,
)


class FirmwareAxiDmaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = compile_firmware_library("axi_dma")
        cls.lib.bcuda_axi_dma_init.argtypes = [ctypes.POINTER(BcudaAxiDma), ctypes.c_size_t, ctypes.c_uint32]
        cls.lib.bcuda_axi_dma_submit_gemm.restype = ctypes.c_int
        cls.lib.bcuda_axi_dma_poll_done.restype = ctypes.c_int

    def test_bcuda_axi_dma_init_sets_register_base_and_timeout(self):
        dma = BcudaAxiDma()
        self.lib.bcuda_axi_dma_init(ctypes.byref(dma), 0xA0000000, 100)
        self.assertEqual(dma.register_base, 0xA0000000)
        self.assertEqual(dma.timeout_cycles, 100)

    def test_bcuda_axi_dma_submit_gemm_accepts_int8_int32_path(self):
        dma = BcudaAxiDma(0, 100, 0)
        desc = BcudaGemmDesc(1, 2, 3, 2, 2, 2, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        A = BcudaBuffer(1, 0x1000, None, 4, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(2, 2, 1, 1), 1)
        B = BcudaBuffer(2, 0x2000, None, 4, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(2, 2, 1, 1), 1)
        C = BcudaBuffer(3, 0x3000, None, 16, BC_DTYPE_INT32, 1, 4, 2, (ctypes.c_uint32 * 4)(2, 2, 1, 1), 1)
        self.assertEqual(self.lib.bcuda_axi_dma_submit_gemm(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(A), ctypes.byref(B), ctypes.byref(C)), BC_STATUS_OK)
        self.assertNotEqual(dma.last_control_word, 0)

    def test_bcuda_axi_dma_submit_gemm_rejects_unsupported_dtype(self):
        dma = BcudaAxiDma(0, 100, 0)
        desc = BcudaGemmDesc(1, 2, 3, 2, 2, 2, BC_DTYPE_INT32, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        buffer = BcudaBuffer()
        self.assertEqual(self.lib.bcuda_axi_dma_submit_gemm(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(buffer), ctypes.byref(buffer), ctypes.byref(buffer)), BC_STATUS_UNSUPPORTED_DTYPE)

    def test_bcuda_axi_dma_poll_done_times_out_when_timeout_is_zero(self):
        dma = BcudaAxiDma(0, 0, 0)
        self.assertEqual(self.lib.bcuda_axi_dma_poll_done(ctypes.byref(dma)), BC_STATUS_EXECUTION_FAILED)


if __name__ == "__main__":
    unittest.main()
