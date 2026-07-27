import ctypes
import unittest

from common import (
    BC_DTYPE_INT4,
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
    BcudaPlDiagnostics,
    BcudaLayerDesc,
    BcudaRequantParams,
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
        cls.lib.bcuda_axi_dma_submit_fused_conv2d.restype = ctypes.c_int
        cls.lib.bcuda_axi_dma_configure_accumulator.restype = ctypes.c_int
        cls.lib.bcuda_axi_dma_configure_memory_io.restype = ctypes.c_int
        cls.lib.bcuda_axi_dma_poll_done.restype = ctypes.c_int
        cls.lib.bcuda_axi_dma_read_diagnostics.argtypes = [
            ctypes.POINTER(BcudaAxiDma),
            ctypes.POINTER(BcudaPlDiagnostics),
        ]
        cls.lib.bcuda_axi_dma_read_diagnostics.restype = ctypes.c_int

    def test_reads_pl_diagnostics_bank(self):
        regs = (ctypes.c_uint32 * 32)()
        for index in range(16, 30):
            regs[index] = index * 3
        regs[30] = 0x04030201
        regs[31] = 0x44494132
        dma = BcudaAxiDma(ctypes.addressof(regs), 8, 0)
        counters = BcudaPlDiagnostics()

        self.assertEqual(
            self.lib.bcuda_axi_dma_read_diagnostics(
                ctypes.byref(dma), ctypes.byref(counters)
            ),
            BC_STATUS_OK,
        )
        self.assertEqual(counters.total_cycles, 48)
        self.assertEqual(counters.cache_wait_cycles, 60)
        self.assertEqual(counters.cache_full_cycles, 87)
        self.assertEqual(counters.max_allocated_tiles, 1)
        self.assertEqual(counters.max_ready_tiles, 2)
        self.assertEqual(counters.allocated_tiles, 3)
        self.assertEqual(counters.ready_tiles, 4)

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

    def test_bcuda_axi_dma_submit_gemm_selects_packed_int4_dual_array_path(self):
        regs = (ctypes.c_uint32 * 16)()
        dma = BcudaAxiDma(ctypes.addressof(regs), 8, 0)
        desc = BcudaGemmDesc(1, 2, 3, 8, 16, 8, BC_DTYPE_INT4, BC_DTYPE_INT4, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        A = BcudaBuffer(1, 0x1000, None, 32, BC_DTYPE_INT4, 1, 2, 2, (ctypes.c_uint32 * 4)(8, 8, 1, 1), 1)
        B = BcudaBuffer(2, 0x2000, None, 64, BC_DTYPE_INT4, 1, 2, 2, (ctypes.c_uint32 * 4)(8, 16, 1, 1), 1)
        C = BcudaBuffer(3, 0x3000, None, 512, BC_DTYPE_INT32, 1, 4, 2, (ctypes.c_uint32 * 4)(8, 16, 1, 1), 1)

        self.assertEqual(self.lib.bcuda_axi_dma_submit_gemm(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(A), ctypes.byref(B), ctypes.byref(C)), BC_STATUS_OK)
        self.assertEqual(dma.last_tile_size, 8)
        self.assertEqual(dma.last_array_count, 2)
        self.assertEqual(regs[5:8], [4, 8, 64])
        self.assertEqual(regs[11:13], [8, 2])
        self.assertEqual(regs[0], 0x101)

    def test_bcuda_axi_dma_submit_tile_load_counts_packed_int4_bytes(self):
        dma = BcudaAxiDma(0, 100, 0, 0, 0)
        desc = BcudaTileLoadDesc(1, 0, 0, 4, 8, 8, BC_DTYPE_INT4, 0)
        source = BcudaBuffer(1, 0x80000000, None, 32, BC_DTYPE_INT4, 1, 2, 2, (ctypes.c_uint32 * 4)(8, 8, 1, 1), 1)
        self.assertEqual(self.lib.bcuda_axi_dma_submit_tile_load(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(source)), BC_STATUS_OK)
        self.assertEqual(dma.last_transfer_bytes, 32)

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

    def test_fused_conv_programs_packed_int4_output_path(self):
        regs = (ctypes.c_uint32 * 16)()
        regs[1] = 0x2
        dma = BcudaAxiDma(ctypes.addressof(regs), 8, 0)
        layer = BcudaLayerDesc()
        layer.n = 1
        layer.c = 8
        layer.h = 8
        layer.w = 8
        layer.out_c = 16
        layer.out_h = 8
        layer.out_w = 8
        layer.kernel_h = 3
        layer.kernel_w = 3
        layer.stride_h = 1
        layer.stride_w = 1
        layer.pad_h = 1
        layer.pad_w = 1
        layer.groups = 1
        activation = BcudaLayerDesc()
        params = BcudaRequantParams(3, 4, 0, -1)
        input_buffer = BcudaBuffer(
            1, 0x1000, None, 256, BC_DTYPE_INT4, 1, 2, 4,
            (ctypes.c_uint32 * 4)(1, 8, 8, 8), 1
        )
        weight_buffer = BcudaBuffer(
            2, 0x2000, None, 576, BC_DTYPE_INT4, 1, 2, 4,
            (ctypes.c_uint32 * 4)(16, 8, 3, 3), 1
        )
        output_buffer = BcudaBuffer(
            3, 0x3000, None, 512, BC_DTYPE_INT4, 1, 2, 4,
            (ctypes.c_uint32 * 4)(1, 16, 8, 8), 1
        )
        status = self.lib.bcuda_axi_dma_submit_fused_conv2d(
            ctypes.byref(dma), ctypes.byref(layer), ctypes.byref(params),
            ctypes.byref(activation), ctypes.byref(input_buffer),
            ctypes.byref(weight_buffer), None, ctypes.byref(output_buffer)
        )
        self.assertEqual(status, BC_STATUS_OK)
        self.assertEqual(regs[0], 0x505)
        self.assertEqual(regs[4], 0x3000)
        self.assertEqual([regs[8], regs[9], regs[10]], [16, 8, 8])

    def test_bcuda_axi_dma_submit_gemm_rejects_unpadded_tile_shape(self):
        dma = BcudaAxiDma(0, 100, 0)
        desc = BcudaGemmDesc(1, 2, 3, 2, 2, 2, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        A = BcudaBuffer(1, 0x1000, None, 4, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(2, 2, 1, 1), 1)
        B = BcudaBuffer(2, 0x2000, None, 4, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(2, 2, 1, 1), 1)
        C = BcudaBuffer(3, 0x3000, None, 16, BC_DTYPE_INT32, 1, 4, 2, (ctypes.c_uint32 * 4)(2, 2, 1, 1), 1)
        self.assertEqual(self.lib.bcuda_axi_dma_submit_gemm(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(A), ctypes.byref(B), ctypes.byref(C)), BC_STATUS_SHAPE_MISMATCH)

    def test_bcuda_axi_dma_submit_gemm_programs_multi_tile_shape(self):
        regs = (ctypes.c_uint32 * 16)()
        dma = BcudaAxiDma(ctypes.addressof(regs), 8, 0)
        desc = BcudaGemmDesc(1, 2, 3, 128, 64, 32, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        A = BcudaBuffer(1, 0x1000, None, 128 * 32, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(128, 32, 1, 1), 1)
        B = BcudaBuffer(2, 0x3000, None, 32 * 64, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(32, 64, 1, 1), 1)
        C = BcudaBuffer(3, 0x4000, None, 128 * 64 * 4, BC_DTYPE_INT32, 1, 4, 2, (ctypes.c_uint32 * 4)(128, 64, 1, 1), 1)

        self.assertEqual(self.lib.bcuda_axi_dma_submit_gemm(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(A), ctypes.byref(B), ctypes.byref(C)), BC_STATUS_OK)
        self.assertEqual([regs[8], regs[9], regs[10]], [128, 64, 32])
        self.assertEqual([regs[5], regs[6], regs[7]], [32, 64, 256])
        self.assertEqual(regs[0], 1)

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
        desc = BcudaGemmDesc(1, 2, 3, 48, 48, 32, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        config = BcudaTiledGemmConfig(16, 1, 0, 0)
        A = BcudaBuffer(1, 0x1000, None, 48 * 32, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(48, 32, 1, 1), 1)
        B = BcudaBuffer(2, 0x2000, None, 32 * 48, BC_DTYPE_INT8, 1, 2, 2, (ctypes.c_uint32 * 4)(32, 48, 1, 1), 1)
        C = BcudaBuffer(3, 0x3000, None, 48 * 48 * 4, BC_DTYPE_INT32, 1, 4, 2, (ctypes.c_uint32 * 4)(48, 48, 1, 1), 1)
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
