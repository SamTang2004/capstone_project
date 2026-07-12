import ctypes
import unittest

from common import (
    BC_CMD_ALLOC,
    BC_DTYPE_INT8,
    BC_DTYPE_INT32,
    BC_STATUS_OK,
    BC_USAGE_ACTIVATION,
    BC_USAGE_OUTPUT,
    BcudaAllocRequest,
    BcudaAllocResponse,
    BcudaAxiDma,
    BcudaExecuteRequest,
    BcudaExecuteResponse,
    BcudaFreeRequest,
    BcudaFreeResponse,
    BcudaGemmDesc,
    BcudaGraphRuntime,
    BcudaReadRequest,
    BcudaReadResponse,
    BcudaWriteRequest,
    BcudaWriteResponse,
    alloc_buffer,
    compile_firmware_library,
    header,
    init_buffer_manager,
    tensor_desc,
)


BC_CMD_WRITE = 2
BC_CMD_READ = 3
BC_CMD_EXECUTE = 4
BC_CMD_FREE = 5
BC_OP_GEMM = 1


class FirmwareCommandDispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = compile_firmware_library("command")
        cls.lib.bcuda_handle_alloc.restype = ctypes.c_int
        cls.lib.bcuda_handle_write.restype = ctypes.c_int
        cls.lib.bcuda_handle_read.restype = ctypes.c_int
        cls.lib.bcuda_handle_execute.restype = ctypes.c_int
        cls.lib.bcuda_handle_free.restype = ctypes.c_int

    def setUp(self):
        self.arena = init_buffer_manager(self.lib)

    def test_bcuda_handle_alloc_allocates_and_fills_response(self):
        req = BcudaAllocRequest(header(BC_CMD_ALLOC), 8, tensor_desc(dim0=8), BC_USAGE_ACTIVATION, 64)
        res = BcudaAllocResponse()
        self.assertEqual(self.lib.bcuda_handle_alloc(ctypes.byref(req), ctypes.byref(res)), BC_STATUS_OK)
        self.assertGreater(res.buffer_id, 0)
        self.assertEqual(res.size_bytes, 8)
        self.assertEqual(res.header.status, BC_STATUS_OK)

    def test_bcuda_handle_write_and_read_round_trip(self):
        buffer_id = alloc_buffer(self.lib, 4, tensor_desc(dim0=4))
        payload = (ctypes.c_uint8 * 4)(9, 8, 7, 6)
        write_req = BcudaWriteRequest(header(BC_CMD_WRITE, 2, 4), buffer_id, 0, 0, 4)
        write_res = BcudaWriteResponse()
        self.assertEqual(self.lib.bcuda_handle_write(ctypes.byref(write_req), payload, ctypes.byref(write_res)), BC_STATUS_OK)
        self.assertEqual(write_res.written_bytes, 4)

        out_payload = (ctypes.c_uint8 * 4)()
        read_req = BcudaReadRequest(header(BC_CMD_READ, 3), buffer_id, 0, 0, 4)
        read_res = BcudaReadResponse()
        self.assertEqual(self.lib.bcuda_handle_read(ctypes.byref(read_req), ctypes.byref(read_res), out_payload), BC_STATUS_OK)
        self.assertEqual(read_res.read_bytes, 4)
        self.assertEqual(list(out_payload), [9, 8, 7, 6])

    def test_bcuda_handle_execute_routes_gemm_to_runtime(self):
        dma = BcudaAxiDma(0, 100, 0)
        runtime = BcudaGraphRuntime()
        self.lib.bcuda_graph_runtime_init(ctypes.byref(runtime), ctypes.byref(dma))
        A = alloc_buffer(self.lib, 256, tensor_desc(BC_DTYPE_INT8, 2, 16))
        B = alloc_buffer(self.lib, 256, tensor_desc(BC_DTYPE_INT8, 2, 16))
        C = alloc_buffer(self.lib, 1024, tensor_desc(BC_DTYPE_INT32, 2, 16), usage=BC_USAGE_OUTPUT)
        desc = BcudaGemmDesc(A, B, C, 16, 16, 16, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        req = BcudaExecuteRequest(header(BC_CMD_EXECUTE, 4, ctypes.sizeof(desc)), BC_OP_GEMM, ctypes.sizeof(desc))
        res = BcudaExecuteResponse()
        self.assertEqual(self.lib.bcuda_handle_execute(ctypes.byref(runtime), ctypes.byref(req), ctypes.byref(desc), ctypes.byref(res)), BC_STATUS_OK)
        self.assertEqual(res.execution_status, BC_STATUS_OK)
        self.assertNotEqual(dma.last_control_word, 0)

    def test_bcuda_handle_free_releases_buffer(self):
        buffer_id = alloc_buffer(self.lib, 4, tensor_desc(dim0=4))
        req = BcudaFreeRequest(header(BC_CMD_FREE, 5), buffer_id, 0)
        res = BcudaFreeResponse()
        self.assertEqual(self.lib.bcuda_handle_free(ctypes.byref(req), ctypes.byref(res)), BC_STATUS_OK)
        self.assertEqual(res.buffer_id, buffer_id)


if __name__ == "__main__":
    unittest.main()
