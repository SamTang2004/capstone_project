import ctypes
import unittest

from common import (
    BC_DTYPE_INT8,
    BC_DTYPE_INT32,
    BC_STATUS_INVALID_HANDLE,
    BC_STATUS_OK,
    BC_USAGE_OUTPUT,
    BcudaAxiDma,
    BcudaGemmDesc,
    BcudaGraphExecDesc,
    BcudaGraphRuntime,
    alloc_buffer,
    compile_firmware_library,
    init_buffer_manager,
    tensor_desc,
)


class FirmwareGraphExecutorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = compile_firmware_library("graph")
        cls.lib.bcuda_graph_runtime_init.argtypes = [ctypes.POINTER(BcudaGraphRuntime), ctypes.POINTER(BcudaAxiDma)]
        cls.lib.bcuda_execute_gemm.restype = ctypes.c_int
        cls.lib.bcuda_execute_graph.restype = ctypes.c_int

    def setUp(self):
        self.arena = init_buffer_manager(self.lib)
        self.dma = BcudaAxiDma(0, 100, 0)
        self.runtime = BcudaGraphRuntime()
        self.lib.bcuda_graph_runtime_init(ctypes.byref(self.runtime), ctypes.byref(self.dma))

    def test_bcuda_graph_runtime_init_stores_dma_pointer(self):
        self.assertTrue(bool(self.runtime.dma))

    def test_bcuda_execute_gemm_resolves_buffers_and_submits_dma(self):
        A = alloc_buffer(self.lib, 4, tensor_desc(BC_DTYPE_INT8, 2, 2))
        B = alloc_buffer(self.lib, 4, tensor_desc(BC_DTYPE_INT8, 2, 2))
        C = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT32, 2, 2), usage=BC_USAGE_OUTPUT)
        desc = BcudaGemmDesc(A, B, C, 2, 2, 2, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        self.assertEqual(self.lib.bcuda_execute_gemm(ctypes.byref(self.runtime), ctypes.byref(desc)), BC_STATUS_OK)
        self.assertNotEqual(self.dma.last_control_word, 0)

    def test_bcuda_execute_gemm_rejects_invalid_buffer_handle(self):
        desc = BcudaGemmDesc(999, 2, 3, 2, 2, 2, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        self.assertEqual(self.lib.bcuda_execute_gemm(ctypes.byref(self.runtime), ctypes.byref(desc)), BC_STATUS_INVALID_HANDLE)

    def test_bcuda_execute_graph_validates_descriptor_buffers_exist(self):
        model = alloc_buffer(self.lib, 64)
        inp = alloc_buffer(self.lib, 4)
        out = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT32, 1, 4), usage=BC_USAGE_OUTPUT)
        desc = BcudaGraphExecDesc(model, inp, out, 0, 0)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(desc)), BC_STATUS_OK)


if __name__ == "__main__":
    unittest.main()
