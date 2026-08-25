import ctypes
import unittest

from common import (
    BC_DTYPE_INT8,
    BC_DTYPE_INT32,
    BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8,
    BC_LAYOUT_TRANSFORMER_DYNAMIC_B_TILED_INT8,
    BC_LAYOUT_ROW_MAJOR,
    BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT4,
    BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT8,
    BC_STATUS_EXECUTION_FAILED,
    BC_STATUS_OK,
    BC_STATUS_SHAPE_MISMATCH,
    BC_STATUS_UNSUPPORTED_DTYPE,
    BcudaAxiDma,
    BcudaAccumulatorConfig,
    BcudaBuffer,
    BcudaGemmDesc,
    BcudaMemoryIoConfig,
    BcudaRequantDesc,
    BcudaRequantParams,
    BcudaLayerDesc,
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
        cls.lib.bcuda_axi_dma_submit_transformer_gemm.restype = ctypes.c_int
        cls.lib.bcuda_axi_dma_submit_transformer_gemm_requant.restype = ctypes.c_int
        cls.lib.bcuda_axi_dma_submit_transformer_requantize_pack.restype = ctypes.c_int
        cls.lib.bcuda_axi_dma_submit_transformer_b_pack.restype = ctypes.c_int
        cls.lib.bcuda_axi_dma_submit_heatvit_token_prune.restype = ctypes.c_int
        cls.lib.bcuda_axi_dma_submit_conv2d.restype = ctypes.c_int
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

    def test_submit_transformer_gemm_programs_packed_w4_contract(self):
        regs = (ctypes.c_uint32 * 64)()
        dma = BcudaAxiDma(ctypes.addressof(regs), 8, 0)
        desc = BcudaGemmDesc(1, 2, 3, 3, 2, 2, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        A = BcudaBuffer(1, 0x1000, None, 16, BC_DTYPE_INT8, 1, 1, 1, (ctypes.c_uint32 * 4)(16, 1, 1, 1), 1)
        B = BcudaBuffer(2, 0x2000, None, 8, BC_DTYPE_INT8, 1, 1, 1, (ctypes.c_uint32 * 4)(8, 1, 1, 1), 1)
        C = BcudaBuffer(3, 0x3000, None, 24, BC_DTYPE_INT32, 1, 4, 2, (ctypes.c_uint32 * 4)(3, 2, 1, 1), 1)
        result = self.lib.bcuda_axi_dma_submit_transformer_gemm(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(A), ctypes.byref(B), ctypes.byref(C), 1)
        self.assertEqual(result, BC_STATUS_OK)
        self.assertEqual(regs[0], 0x801)
        self.assertEqual([regs[8], regs[9], regs[10]], [3, 2, 2])
        self.assertEqual([regs[5], regs[6], regs[7]], [16, 8, 8])
        self.assertEqual([regs[11], regs[12]], [8, 1])
        self.assertEqual([dma.last_m_tiles, dma.last_n_tiles, dma.last_k_tiles], [1, 1, 2])

    def test_submit_transformer_requantize_pack_programs_fused_operation(self):
        regs = (ctypes.c_uint32 * 64)()
        dma = BcudaAxiDma(ctypes.addressof(regs), 8, 0)
        desc = BcudaRequantDesc(1, 2, 9, 7, 2, 0, -1, 0)
        source = BcudaBuffer(1, 0x1000, None, 36, BC_DTYPE_INT32, 1, 4, 2, (ctypes.c_uint32 * 4)(3, 3, 1, 1), 1)
        output = BcudaBuffer(2, 0x2000, None, 24, BC_DTYPE_INT8, 1, 1, 1, (ctypes.c_uint32 * 4)(24, 1, 1, 1), 1)
        result = self.lib.bcuda_axi_dma_submit_transformer_requantize_pack(ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(source), ctypes.byref(output), 3, 3)
        self.assertEqual(result, BC_STATUS_OK)
        self.assertEqual(regs[0], 0x1001)
        self.assertEqual([regs[2], regs[4], regs[5]], [0x1000, 0x2000, 12])
        self.assertEqual([regs[8], regs[9], regs[13], regs[14], regs[15]], [3, 3, 7, 2, 0xffffffff])

    def test_submit_transformer_gemm_requant_programs_direct_w4a8_output(self):
        regs = (ctypes.c_uint32 * 64)()
        dma = BcudaAxiDma(ctypes.addressof(regs), 8, 0)
        desc = BcudaGemmDesc(1, 2, 3, 3, 2, 2, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT8, 1, 1, BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8, 0, 0)
        params = BcudaRequantParams(7, 2, 0, -3)
        A = BcudaBuffer(1, 0x1000, None, 16, BC_DTYPE_INT8, BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8, 1, 1, (ctypes.c_uint32 * 4)(16, 1, 1, 1), 1)
        B = BcudaBuffer(2, 0x2000, None, 8, BC_DTYPE_INT8, 1, 1, 1, (ctypes.c_uint32 * 4)(8, 1, 1, 1), 1)
        bias = BcudaBuffer(3, 0x2800, None, 8, BC_DTYPE_INT32, 1, 4, 1, (ctypes.c_uint32 * 4)(2, 1, 1, 1), 1)
        C = BcudaBuffer(4, 0x3000, None, 16, BC_DTYPE_INT8, BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8, 1, 1, (ctypes.c_uint32 * 4)(16, 1, 1, 1), 1)
        result = self.lib.bcuda_axi_dma_submit_transformer_gemm_requant(
            ctypes.byref(dma), ctypes.byref(desc), ctypes.byref(params),
            ctypes.byref(A), ctypes.byref(B), ctypes.byref(bias), ctypes.byref(C), 1,
        )
        self.assertEqual(result, BC_STATUS_OK)
        self.assertEqual(regs[0], 0x100801)
        self.assertEqual([regs[2], regs[3], regs[4], regs[6]], [0x1000, 0x2000, 0x3000, 0x2800])
        self.assertEqual([regs[8], regs[9], regs[10]], [3, 2, 2])
        self.assertEqual([regs[13], regs[14], regs[15]], [7, 2, 0xfffffffd])

    def test_submit_transformer_b_pack_programs_transpose_operation(self):
        regs = (ctypes.c_uint32 * 64)()
        dma = BcudaAxiDma(ctypes.addressof(regs), 8, 0)
        source = BcudaBuffer(1, 0x1000, None, 16, BC_DTYPE_INT8, BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8, 2, 2, (ctypes.c_uint32 * 4)(3, 2, 1, 1), 1)
        output = BcudaBuffer(2, 0x2000, None, 16, BC_DTYPE_INT8, BC_LAYOUT_TRANSFORMER_DYNAMIC_B_TILED_INT8, 2, 2, (ctypes.c_uint32 * 4)(2, 3, 1, 1), 1)
        result = self.lib.bcuda_axi_dma_submit_transformer_b_pack(ctypes.byref(dma), ctypes.byref(source), ctypes.byref(output), 2, 3, 1, 0)
        self.assertEqual(result, BC_STATUS_OK)
        self.assertEqual(regs[0], 0x6001)
        self.assertEqual([regs[2], regs[4], regs[8], regs[9]], [0x1000, 0x2000, 2, 3])
        self.assertEqual([dma.last_m_tiles, dma.last_n_tiles, dma.last_k_tiles], [1, 1, 2])

    def test_submit_heatvit_token_prune_programs_ddr_operation(self):
        regs = (ctypes.c_uint32 * 64)()
        dma = BcudaAxiDma(ctypes.addressof(regs), 8, 0)
        inp = BcudaBuffer(1, 0x1000, None, 197 * 192, BC_DTYPE_INT8, BC_LAYOUT_ROW_MAJOR, 2, 2, (ctypes.c_uint32 * 4)(197, 192, 1, 1), 1)
        weight = BcudaBuffer(2, 0x2000, None, 192 * 8, BC_DTYPE_INT8, BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT8, 1, 2, (ctypes.c_uint32 * 4)(192, 1, 1, 1), 1)
        out = BcudaBuffer(3, 0x3000, None, 168 * 192, BC_DTYPE_INT8, BC_LAYOUT_ROW_MAJOR, 4, 2, (ctypes.c_uint32 * 4)(168, 192, 1, 1), 1)
        result = self.lib.bcuda_axi_dma_submit_heatvit_token_prune(
            ctypes.byref(dma), ctypes.byref(inp), ctypes.byref(weight), ctypes.byref(out), 197, 168, 192, 0,
        )
        self.assertEqual(result, BC_STATUS_OK)
        self.assertEqual(regs[0], 0x00800001)
        self.assertEqual([regs[2], regs[3], regs[4]], [0x1000, 0x2000, 0x3000])
        self.assertEqual([regs[8], regs[9], regs[10]], [197, 168, 192])
        self.assertEqual(regs[13], (65536 + 15) // 30)
        self.assertEqual([dma.last_m_tiles, dma.last_n_tiles, dma.last_k_tiles], [197, 168, 192])

    def test_submit_heatvit_token_prune_supports_tiled_w4_scorer(self):
        dma = BcudaAxiDma(0, 8, 0)
        inp = BcudaBuffer(1, 0x1000, None, 5 * 8, BC_DTYPE_INT8, BC_LAYOUT_ROW_MAJOR, 2, 2, (ctypes.c_uint32 * 4)(5, 8, 1, 1), 1)
        weight = BcudaBuffer(2, 0x2000, None, 8 * 4, BC_DTYPE_INT8, BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT4, 1, 2, (ctypes.c_uint32 * 4)(8, 1, 1, 1), 1)
        out = BcudaBuffer(3, 0x3000, None, 3 * 8, BC_DTYPE_INT8, BC_LAYOUT_ROW_MAJOR, 4, 2, (ctypes.c_uint32 * 4)(3, 8, 1, 1), 1)
        result = self.lib.bcuda_axi_dma_submit_heatvit_token_prune(
            ctypes.byref(dma), ctypes.byref(inp), ctypes.byref(weight), ctypes.byref(out), 5, 3, 8, 1,
        )
        self.assertEqual(result, BC_STATUS_OK)
        self.assertEqual(dma.last_control_word, 0x00800801)

    def test_bcuda_axi_dma_submit_conv2d_programs_presliced_weight_stride(self):
        regs = (ctypes.c_uint32 * 16)()
        dma = BcudaAxiDma(ctypes.addressof(regs), 8, 0)
        layer = BcudaLayerDesc()
        layer.n, layer.c, layer.h, layer.w = 1, 1, 3, 3
        layer.out_c, layer.out_h, layer.out_w = 1, 2, 2
        layer.kernel_h, layer.kernel_w = 2, 2
        layer.stride_h = layer.stride_w = 1
        layer.groups, layer.flags = 1, 1
        inp = BcudaBuffer(1, 0x1000, None, 9, BC_DTYPE_INT8, 2, 4, 4, (ctypes.c_uint32 * 4)(1, 1, 3, 3), 1)
        weight = BcudaBuffer(2, 0x2000, None, 256, BC_DTYPE_INT8, 4, 4, 4, (ctypes.c_uint32 * 4)(1, 1, 16, 16), 1)
        out = BcudaBuffer(3, 0x3000, None, 16, BC_DTYPE_INT32, 2, 4, 4, (ctypes.c_uint32 * 4)(1, 1, 2, 2), 1)
        self.assertEqual(self.lib.bcuda_axi_dma_submit_conv2d(ctypes.byref(dma), ctypes.byref(layer), ctypes.byref(inp), ctypes.byref(weight), None, ctypes.byref(out)), BC_STATUS_OK)
        self.assertEqual(regs[6], 3)
        self.assertEqual(regs[0], 0x405)

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
