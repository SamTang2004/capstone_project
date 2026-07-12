import ctypes
import unittest

from common import (
    BC_DTYPE_INT8,
    BC_DTYPE_INT32,
    BC_STATUS_EXECUTION_FAILED,
    BC_STATUS_OK,
    BC_STATUS_SHAPE_MISMATCH,
    BC_STATUS_UNSUPPORTED_DTYPE,
    BcudaAxiDma,
    BcudaAccumulatorConfig,
    BcudaBuffer,
    BcudaGemmDesc,
    BcudaMemoryIoConfig,
    BcudaTileLoadDesc,
    BcudaTiledGemmConfig,
    compile_firmware_library,
)


class FirmwareAxiDmaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = compile_firmware_library("axi_dma")
        cls.lib.bcuda_axi_dma_init.argtypes = [ctypes.POINTER(BcudaAxiDma), ctypes.c_size_t, ctypes.c_uint32]
        cls.lib.bcuda_axi_dma_submit_gemm.restype = ctypes.c_int
        cls.lib.bcuda_axi_dma_submit_tile_load.restype = ctypes.c_int
        cls.lib.bcuda_axi_dma_submit_tiled_gemm.restype = ctypes.c_int
        cls.lib.bcuda_axi_dma_configure_accumulator.restype = ctypes.c_int
        cls.lib.bcuda_axi_dma_configure_memory_io.restype = ctypes.c_int
        cls.lib.bcuda_axi_dma_poll_done.restype = ctypes.c_int

    def test_bcuda_axi_dma_init_sets_register_base_and_timeout(self):
        dma = BcudaAxiDma()
        self.lib.bcuda_axi_dma_init(ctypes.byref(dma), 0xA0000000, 100)
        self.assertEqual(dma.register_base, 0xA0000000)
        self.assertEqual(dma.timeout_cycles, 100)
        self.assertEqual(dma.last_source_address, 0)
        self.assertEqual(dma.last_transfer_bytes, 0)

    def test_bcuda_axi_dma_submit_gemm_accepts_int8_int32_path(self):
        dma = BcudaAxiDma(0, 100, 0)
        desc = BcudaGemmDesc(1, 2, 3, 16, 16, 16, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        A = BcudaBuffer(1, 0x1000, None, 256, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(16, 16, 1, 1), 1)
        B = BcudaBuffer(2, 0x2000, None, 256, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(16, 16, 1, 1), 1)
        C = BcudaBuffer(3, 0x3000, None, 1024, BC_DTYPE_INT32, 1, 4, 2, (ctypes.c_uint32 * 4)(16, 16, 1, 1), 1)
        self.assertEqual(self.lib.bcuda_axi_dma_submit_gemm(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(A), ctypes.byref(B), ctypes.byref(C)), BC_STATUS_OK)
        self.assertNotEqual(dma.last_control_word, 0)

    def test_bcuda_axi_dma_submit_gemm_programs_final_pl_register_map(self):
        regs = (ctypes.c_uint32 * 16)()
        dma = BcudaAxiDma(ctypes.addressof(regs), 8, 0)
        desc = BcudaGemmDesc(1, 2, 3, 16, 16, 16, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        A = BcudaBuffer(1, 0x1000, None, 256, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(16, 16, 1, 1), 1)
        B = BcudaBuffer(2, 0x2000, None, 256, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(16, 16, 1, 1), 1)
        C = BcudaBuffer(3, 0x3000, None, 1024, BC_DTYPE_INT32, 1, 4, 2, (ctypes.c_uint32 * 4)(16, 16, 1, 1), 1)

        self.assertEqual(self.lib.bcuda_axi_dma_submit_gemm(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(A), ctypes.byref(B), ctypes.byref(C)), BC_STATUS_OK)
        self.assertEqual(regs[0], 0x1)
        self.assertEqual(regs[2], 0x1000)
        self.assertEqual(regs[3], 0x2000)
        self.assertEqual(regs[4], 0x3000)
        self.assertEqual(regs[5], 16)
        self.assertEqual(regs[6], 16)
        self.assertEqual(regs[7], 64)
        self.assertEqual(regs[8], 16)
        self.assertEqual(regs[9], 16)
        self.assertEqual(regs[10], 16)
        self.assertEqual(regs[11], 16)
        self.assertEqual(regs[12], 1)

        regs[1] = 0x2
        self.assertEqual(self.lib.bcuda_axi_dma_poll_done(ctypes.byref(dma)), BC_STATUS_OK)
        self.assertEqual(regs[0], 0)

    def test_bcuda_axi_dma_submit_gemm_rejects_non_final_tile_shape(self):
        dma = BcudaAxiDma(0, 100, 0)
        desc = BcudaGemmDesc(1, 2, 3, 2, 2, 2, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        A = BcudaBuffer(1, 0x1000, None, 4, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(2, 2, 1, 1), 1)
        B = BcudaBuffer(2, 0x2000, None, 4, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(2, 2, 1, 1), 1)
        C = BcudaBuffer(3, 0x3000, None, 16, BC_DTYPE_INT32, 1, 4, 2, (ctypes.c_uint32 * 4)(2, 2, 1, 1), 1)
        self.assertEqual(self.lib.bcuda_axi_dma_submit_gemm(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(A), ctypes.byref(B), ctypes.byref(C)), BC_STATUS_SHAPE_MISMATCH)

    def test_bcuda_axi_dma_submit_gemm_rejects_unsupported_dtype(self):
        dma = BcudaAxiDma(0, 100, 0)
        desc = BcudaGemmDesc(1, 2, 3, 16, 16, 16, BC_DTYPE_INT32, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        buffer = BcudaBuffer()
        self.assertEqual(self.lib.bcuda_axi_dma_submit_gemm(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(buffer), ctypes.byref(buffer), ctypes.byref(buffer)), BC_STATUS_UNSUPPORTED_DTYPE)

    def test_bcuda_axi_dma_submit_tile_load_records_source_window(self):
        dma = BcudaAxiDma(0, 100, 0, 0, 0)
        desc = BcudaTileLoadDesc(1, 1, 0x65, 32, 16, 16, BC_DTYPE_INT8, 0)
        source = BcudaBuffer(1, 0x80000000, None, 1024, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(32, 32, 1, 1), 1)
        self.assertEqual(self.lib.bcuda_axi_dma_submit_tile_load(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(source)), BC_STATUS_OK)
        self.assertEqual(dma.last_source_address, 0x80000065)
        self.assertEqual(dma.last_transfer_bytes, 256)
        self.assertNotEqual(dma.last_control_word, 0)

    def test_bcuda_axi_dma_submit_tile_load_rejects_out_of_range_tile(self):
        dma = BcudaAxiDma(0, 100, 0, 0, 0)
        desc = BcudaTileLoadDesc(1, 0, 1000, 32, 16, 16, BC_DTYPE_INT8, 0)
        source = BcudaBuffer(1, 0x80000000, None, 1024, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(32, 32, 1, 1), 1)
        self.assertEqual(self.lib.bcuda_axi_dma_submit_tile_load(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(source)), BC_STATUS_SHAPE_MISMATCH)

    def test_bcuda_axi_dma_submit_tile_load_rejects_non_int8(self):
        dma = BcudaAxiDma(0, 100, 0, 0, 0)
        desc = BcudaTileLoadDesc(1, 0, 0, 32, 16, 16, BC_DTYPE_INT32, 0)
        source = BcudaBuffer(1, 0x80000000, None, 1024, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(32, 32, 1, 1), 1)
        self.assertEqual(self.lib.bcuda_axi_dma_submit_tile_load(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(source)), BC_STATUS_UNSUPPORTED_DTYPE)

    def test_bcuda_axi_dma_submit_tiled_gemm_records_controller_schedule(self):
        dma = BcudaAxiDma()
        desc = BcudaGemmDesc(1, 2, 3, 35, 41, 27, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        config = BcudaTiledGemmConfig(16, 1, 0, 0)
        A = BcudaBuffer(1, 0x1000, None, 35 * 27, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(35, 27, 1, 1), 1)
        B = BcudaBuffer(2, 0x2000, None, 27 * 41, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(27, 41, 1, 1), 1)
        C = BcudaBuffer(3, 0x3000, None, 35 * 41 * 4, BC_DTYPE_INT32, 1, 4, 2, (ctypes.c_uint32 * 4)(35, 41, 1, 1), 1)
        self.assertEqual(self.lib.bcuda_axi_dma_submit_tiled_gemm(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(config), ctypes.byref(A), ctypes.byref(B), ctypes.byref(C)), BC_STATUS_OK)
        self.assertEqual(dma.last_tile_size, 16)
        self.assertEqual(dma.last_array_count, 1)
        self.assertEqual(dma.last_m_tiles, 3)
        self.assertEqual(dma.last_n_tiles, 3)
        self.assertEqual(dma.last_k_tiles, 2)
        self.assertEqual(dma.last_n_groups, 3)
        self.assertEqual(dma.last_schedule_steps, 18)
        self.assertNotEqual(dma.last_control_word, 0)

    def test_bcuda_axi_dma_submit_tiled_gemm_rejects_invalid_shape(self):
        dma = BcudaAxiDma()
        desc = BcudaGemmDesc(1, 2, 3, 35, 41, 27, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        config = BcudaTiledGemmConfig(0, 2, 0, 0)
        A = BcudaBuffer(1, 0x1000, None, 1024, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(1, 1, 1, 1), 1)
        B = BcudaBuffer(2, 0x2000, None, 1024, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(1, 1, 1, 1), 1)
        C = BcudaBuffer(3, 0x3000, None, 4096, BC_DTYPE_INT32, 1, 4, 2, (ctypes.c_uint32 * 4)(1, 1, 1, 1), 1)
        self.assertEqual(self.lib.bcuda_axi_dma_submit_tiled_gemm(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(config), ctypes.byref(A), ctypes.byref(B), ctypes.byref(C)), BC_STATUS_SHAPE_MISMATCH)

    def test_bcuda_axi_dma_submit_tiled_gemm_rejects_unsupported_dtype(self):
        dma = BcudaAxiDma()
        desc = BcudaGemmDesc(1, 2, 3, 35, 41, 27, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        config = BcudaTiledGemmConfig(16, 2, 0, 0)
        A = BcudaBuffer(1, 0x1000, None, 1024, BC_DTYPE_INT32, 1, 2, 2, (ctypes.c_uint32 * 4)(1, 1, 1, 1), 1)
        B = BcudaBuffer(2, 0x2000, None, 1024, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(1, 1, 1, 1), 1)
        C = BcudaBuffer(3, 0x3000, None, 4096, BC_DTYPE_INT32, 1, 4, 2, (ctypes.c_uint32 * 4)(1, 1, 1, 1), 1)
        self.assertEqual(self.lib.bcuda_axi_dma_submit_tiled_gemm(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(config), ctypes.byref(A), ctypes.byref(B), ctypes.byref(C)), BC_STATUS_UNSUPPORTED_DTYPE)

    def test_bcuda_axi_dma_configure_accumulator_records_tile_entries(self):
        dma = BcudaAxiDma()
        config = BcudaAccumulatorConfig(256, 1, 0, 0)
        self.assertEqual(self.lib.bcuda_axi_dma_configure_accumulator(ctypes.byref(dma), ctypes.byref(config)), BC_STATUS_OK)
        self.assertEqual(dma.last_accumulator_entries, 256)
        self.assertNotEqual(dma.last_control_word, 0)

    def test_bcuda_axi_dma_configure_accumulator_rejects_too_many_entries(self):
        dma = BcudaAxiDma()
        config = BcudaAccumulatorConfig(257, 1, 0, 0)
        self.assertEqual(self.lib.bcuda_axi_dma_configure_accumulator(ctypes.byref(dma), ctypes.byref(config)), BC_STATUS_SHAPE_MISMATCH)

    def test_bcuda_axi_dma_configure_memory_io_records_ddr_window(self):
        dma = BcudaAxiDma()
        config = BcudaMemoryIoConfig(0x18040000, 4096, 0x3)
        self.assertEqual(self.lib.bcuda_axi_dma_configure_memory_io(ctypes.byref(dma), ctypes.byref(config)), BC_STATUS_OK)
        self.assertEqual(dma.last_memory_base, 0x18040000)
        self.assertEqual(dma.last_memory_bytes, 4096)
        self.assertEqual(dma.last_memory_io_flags, 0x3)
        self.assertNotEqual(dma.last_control_word, 0)

    def test_bcuda_axi_dma_configure_memory_io_rejects_unaligned_window(self):
        dma = BcudaAxiDma()
        config = BcudaMemoryIoConfig(0x18040002, 4096, 0)
        self.assertEqual(self.lib.bcuda_axi_dma_configure_memory_io(ctypes.byref(dma), ctypes.byref(config)), BC_STATUS_SHAPE_MISMATCH)

    def test_bcuda_axi_dma_poll_done_times_out_when_timeout_is_zero(self):
        dma = BcudaAxiDma(0, 0, 0)
        self.assertEqual(self.lib.bcuda_axi_dma_poll_done(ctypes.byref(dma)), BC_STATUS_EXECUTION_FAILED)


if __name__ == "__main__":
    unittest.main()
