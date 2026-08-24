import ctypes
import struct
import unittest

from common import (
    BC_OP_GEMM,
    BC_OP_ACTIVATION,
    BC_OP_CONV2D_GEMM,
    BC_OP_CONCAT,
    BC_OP_NMS,
    BC_OP_POOL,
    BC_OP_RESHAPE,
    BC_OP_REQUANTIZE,
    BC_OP_UPSAMPLE,
    BC_OP_LAYERNORM,
    BC_OP_GELU,
    BC_OP_SOFTMAX,
    BC_OP_RESIDUAL_ADD,
    BC_OP_ACTIVATION_B_PACK,
    BC_OP_TOKEN_CONCAT,
    BC_OP_TOKEN_EMBEDDING,
    BCUDA_DIAG_ALL,
    BCUDA_DIAGNOSTICS_MAGIC,
    BCUDA_DIAGNOSTICS_VERSION,
    BCUDA_MODEL_MAGIC,
    BCUDA_MODEL_VERSION,
    BC_DTYPE_INT8,
    BC_DTYPE_INT16,
    BC_DTYPE_INT32,
    BC_LAYOUT_ROW_MAJOR,
    BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8,
    BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT8,
    BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT4,
    BC_LAYOUT_TRANSFORMER_DYNAMIC_B_TILED_INT8,
    BC_STATUS_INVALID_COMMAND,
    BC_STATUS_INVALID_HANDLE,
    BC_STATUS_OK,
    BC_USAGE_DESCRIPTOR,
    BC_USAGE_OUTPUT,
    BcudaAxiDma,
    BcudaDiagnosticsHeader,
    BcudaDiagnosticsRecord,
    BcudaGemmDesc,
    BcudaGraphExecDesc,
    BcudaGraphRuntime,
    BcudaLayerDesc,
    BcudaModelBufferDesc,
    BcudaModelDesc,
    alloc_buffer,
    compile_firmware_library,
    init_buffer_manager,
    read_buffer,
    tensor_desc,
    write_buffer,
)

MODEL_DESC_STRUCT = struct.Struct("<IIIIIIIIIIIIII")
BUFFER_DESC_STRUCT = struct.Struct("<IIIIIIII")
LAYER_DESC_STRUCT = struct.Struct("<IIIIIIIIIIIIIIIIIIIIIIII")


def build_single_gemm_model_descriptor(input_id, weight_id, output_id, op_type=BC_OP_GEMM):
    buffer_table_offset = MODEL_DESC_STRUCT.size
    layer_desc_offset = buffer_table_offset + 3 * BUFFER_DESC_STRUCT.size
    model = MODEL_DESC_STRUCT.pack(
        BCUDA_MODEL_MAGIC,
        BCUDA_MODEL_VERSION,
        1,
        layer_desc_offset,
        3,
        buffer_table_offset,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    buffers = b"".join(
        [
            BUFFER_DESC_STRUCT.pack(input_id, BC_DTYPE_INT8, 1, 2, 256, 2, 16, 16),
            BUFFER_DESC_STRUCT.pack(weight_id, BC_DTYPE_INT8, 1, 1, 256, 2, 16, 16),
            BUFFER_DESC_STRUCT.pack(output_id, BC_DTYPE_INT32, 1, 4, 1024, 2, 16, 16),
        ]
    )
    layer = LAYER_DESC_STRUCT.pack(
        op_type,
        0,
        0,
        0,
        2,
        1,
        0,
        0,
        16,
        16,
        0,
        0,
        16,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
    )
    return model + buffers + layer


def pack_model_descriptor(buffer_entries, layer_entries, input_index=0, output_index=None):
    output_index = len(buffer_entries) - 1 if output_index is None else output_index
    buffer_table_offset = MODEL_DESC_STRUCT.size
    layer_desc_offset = buffer_table_offset + len(buffer_entries) * BUFFER_DESC_STRUCT.size
    model = MODEL_DESC_STRUCT.pack(
        BCUDA_MODEL_MAGIC,
        BCUDA_MODEL_VERSION,
        len(layer_entries),
        layer_desc_offset,
        len(buffer_entries),
        buffer_table_offset,
        input_index,
        output_index,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    buffers = b"".join(BUFFER_DESC_STRUCT.pack(*entry) for entry in buffer_entries)
    layers = b"".join(LAYER_DESC_STRUCT.pack(*entry) for entry in layer_entries)
    return model + buffers + layers


def layer_entry(
    op_type,
    input0=0,
    input1=0,
    input2=0,
    output=1,
    weight=0,
    bias=0,
    scale=0,
    n=0,
    c=0,
    h=0,
    w=0,
    out_c=0,
    out_h=0,
    out_w=0,
    kernel_h=0,
    kernel_w=0,
    stride_h=0,
    stride_w=0,
    pad_h=0,
    pad_w=0,
    groups=1,
    flags=0,
):
    return (
        op_type,
        input0,
        input1,
        input2,
        output,
        weight,
        bias,
        scale,
        n,
        c,
        h,
        w,
        out_c,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        groups,
        flags,
        0,
    )


def int32_bytes(values):
    return b"".join(int(value).to_bytes(4, "little", signed=True) for value in values)


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

    def test_firmware_and_python_il_struct_sizes_match(self):
        self.assertEqual(ctypes.sizeof(BcudaModelDesc), MODEL_DESC_STRUCT.size)
        self.assertEqual(ctypes.sizeof(BcudaModelBufferDesc), BUFFER_DESC_STRUCT.size)
        self.assertEqual(ctypes.sizeof(BcudaLayerDesc), LAYER_DESC_STRUCT.size)

    def test_bcuda_execute_gemm_resolves_buffers_and_submits_dma(self):
        A = alloc_buffer(self.lib, 256, tensor_desc(BC_DTYPE_INT8, 2, 16, 16))
        B = alloc_buffer(self.lib, 256, tensor_desc(BC_DTYPE_INT8, 2, 16, 16))
        C = alloc_buffer(self.lib, 1024, tensor_desc(BC_DTYPE_INT32, 2, 16, 16), usage=BC_USAGE_OUTPUT)
        desc = BcudaGemmDesc(A, B, C, 16, 16, 16, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        self.assertEqual(self.lib.bcuda_execute_gemm(ctypes.byref(self.runtime), ctypes.byref(desc)), BC_STATUS_OK)
        self.assertNotEqual(self.dma.last_control_word, 0)

    def test_bcuda_execute_gemm_pads_irregular_shape_for_pl(self):
        self.arena = init_buffer_manager(self.lib, arena_size=65536)
        A = alloc_buffer(self.lib, 37 * 29, tensor_desc(BC_DTYPE_INT8, 2, 37, 29))
        B = alloc_buffer(self.lib, 29 * 53, tensor_desc(BC_DTYPE_INT8, 2, 29, 53))
        C = alloc_buffer(self.lib, 37 * 53 * 4, tensor_desc(BC_DTYPE_INT32, 2, 37, 53), usage=BC_USAGE_OUTPUT)
        desc = BcudaGemmDesc(A, B, C, 37, 53, 29, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)

        self.assertEqual(self.lib.bcuda_execute_gemm(ctypes.byref(self.runtime), ctypes.byref(desc)), BC_STATUS_OK)
        self.assertEqual(self.dma.last_transfer_bytes, 48 * 32 + 32 * 64 + 48 * 64 * 4)
        self.assertNotEqual(self.dma.last_control_word, 0)

    def test_bcuda_execute_gemm_rejects_invalid_buffer_handle(self):
        desc = BcudaGemmDesc(999, 2, 3, 2, 2, 2, BC_DTYPE_INT8, BC_DTYPE_INT8, BC_DTYPE_INT32, 1, 1, 1, 0, 0)
        self.assertEqual(self.lib.bcuda_execute_gemm(ctypes.byref(self.runtime), ctypes.byref(desc)), BC_STATUS_INVALID_HANDLE)

    def test_bcuda_execute_graph_validates_descriptor_buffers_exist(self):
        model = alloc_buffer(self.lib, 64)
        inp = alloc_buffer(self.lib, 4)
        out = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT32, 1, 4), usage=BC_USAGE_OUTPUT)
        desc = BcudaGraphExecDesc(model, inp, out, 0, 0)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(desc)), BC_STATUS_INVALID_COMMAND)

    def test_bcuda_execute_graph_parses_descriptor_and_executes_gemm_layer(self):
        A = alloc_buffer(self.lib, 256, tensor_desc(BC_DTYPE_INT8, 2, 16, 16))
        B = alloc_buffer(self.lib, 256, tensor_desc(BC_DTYPE_INT8, 2, 16, 16))
        C = alloc_buffer(self.lib, 1024, tensor_desc(BC_DTYPE_INT32, 2, 16, 16), usage=BC_USAGE_OUTPUT)
        descriptor = build_single_gemm_model_descriptor(A, B, C)
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        desc = BcudaGraphExecDesc(model, A, C, 0, 0)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(desc)), BC_STATUS_OK)
        self.assertNotEqual(self.dma.last_control_word, 0)
        self.assertEqual(self.dma.last_tile_size, 16)
        self.assertEqual(self.dma.last_array_count, 1)

    def test_bcuda_execute_graph_routes_tiled_transformer_gemm_w8_and_w4(self):
        for weight_layout, weight_bytes, control_mask in (
            (BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT8, 16, 0),
            (BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT4, 8, 0x800),
        ):
            self.arena = init_buffer_manager(self.lib)
            a = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT8, 2, 3, 2, layout=BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8))
            w = alloc_buffer(self.lib, weight_bytes, tensor_desc(BC_DTYPE_INT8, 2, 2, 3, layout=weight_layout))
            c = alloc_buffer(self.lib, 36, tensor_desc(BC_DTYPE_INT32, 2, 3, 3), usage=BC_USAGE_OUTPUT)
            descriptor = pack_model_descriptor(
                [
                    (a, BC_DTYPE_INT8, BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8, 2, 16, 2, 3, 2),
                    (w, BC_DTYPE_INT8, weight_layout, 1, weight_bytes, 2, 2, 3),
                    (c, BC_DTYPE_INT32, BC_LAYOUT_ROW_MAJOR, 4, 36, 2, 3, 3),
                ],
                [layer_entry(BC_OP_GEMM, input0=0, weight=1, output=2, n=3, c=2, out_c=3)],
                output_index=2,
            )
            model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
            write_buffer(self.lib, model, descriptor)
            self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, a, c, 0, 0))), BC_STATUS_OK)
            self.assertEqual(self.dma.last_tile_size, 8)
            self.assertEqual(self.dma.last_m_tiles, 1)
            self.assertEqual(self.dma.last_n_tiles, 1)
            self.assertEqual(self.dma.last_k_tiles, 2)
            self.assertEqual(self.dma.last_control_word & 0x800, control_mask)

    def test_bcuda_execute_graph_routes_fused_w4a8_gemm_bias_requant(self):
        a = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT8, 2, 3, 2, layout=BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8))
        w = alloc_buffer(self.lib, 8, tensor_desc(BC_DTYPE_INT8, 2, 2, 2, layout=BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT4))
        bias = alloc_buffer(self.lib, 8, tensor_desc(BC_DTYPE_INT32, 1, 2))
        scale = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT32, 1, 4))
        c = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT8, 2, 3, 2, layout=BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, scale, struct.pack("<iIii", 7, 2, 0, -3))
        descriptor = pack_model_descriptor(
            [
                (a, BC_DTYPE_INT8, BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8, 1, 16, 2, 3, 2),
                (w, BC_DTYPE_INT8, BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT4, 1, 8, 2, 2, 2),
                (bias, BC_DTYPE_INT32, BC_LAYOUT_ROW_MAJOR, 4, 8, 1, 2, 1),
                (scale, BC_DTYPE_INT32, BC_LAYOUT_ROW_MAJOR, 4, 16, 1, 4, 1),
                (c, BC_DTYPE_INT8, BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8, 1, 16, 2, 3, 2),
            ],
            [layer_entry(BC_OP_GEMM, input0=0, weight=1, bias=2, scale=3, output=4, n=3, c=2, out_c=2)],
            output_index=4,
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        self.assertEqual(self.lib.bcuda_execute_graph(
            ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, a, c, 0, 0))), BC_STATUS_OK)
        self.assertEqual(self.dma.last_control_word, 0x100801)
        self.assertEqual([self.dma.last_m_tiles, self.dma.last_n_tiles, self.dma.last_k_tiles], [1, 1, 2])

    def test_bcuda_execute_graph_records_layer_diagnostics(self):
        A = alloc_buffer(self.lib, 256, tensor_desc(BC_DTYPE_INT8, 2, 16, 16))
        B = alloc_buffer(self.lib, 256, tensor_desc(BC_DTYPE_INT8, 2, 16, 16))
        C = alloc_buffer(self.lib, 1024, tensor_desc(BC_DTYPE_INT32, 2, 16, 16), usage=BC_USAGE_OUTPUT)
        descriptor = build_single_gemm_model_descriptor(A, B, C)
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)

        report_header = BcudaDiagnosticsHeader(
            BCUDA_DIAGNOSTICS_MAGIC,
            BCUDA_DIAGNOSTICS_VERSION,
            BCUDA_DIAG_ALL,
            1,
            0,
            ctypes.sizeof(BcudaDiagnosticsRecord),
            100_000_000,
            512,
            0,
            0,
        )
        report_data = bytes(report_header) + bytes(ctypes.sizeof(BcudaDiagnosticsRecord))
        report = alloc_buffer(self.lib, len(report_data), tensor_desc(BC_DTYPE_INT8, 1, len(report_data)))
        write_buffer(self.lib, report, report_data)
        desc = BcudaGraphExecDesc(model, A, C, BCUDA_DIAG_ALL, report)

        self.assertEqual(ctypes.sizeof(BcudaDiagnosticsHeader), 48)
        self.assertEqual(ctypes.sizeof(BcudaDiagnosticsRecord), 176)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(desc)), BC_STATUS_OK)
        raw_report = read_buffer(self.lib, report, len(report_data))
        parsed_header = BcudaDiagnosticsHeader.from_buffer_copy(raw_report)
        parsed_record = BcudaDiagnosticsRecord.from_buffer_copy(raw_report, ctypes.sizeof(BcudaDiagnosticsHeader))
        self.assertEqual(parsed_header.layer_count, 1)
        self.assertEqual(parsed_record.op_type, BC_OP_GEMM)
        self.assertEqual(parsed_record.input_bytes, 256)
        self.assertEqual(parsed_record.output_bytes, 1024)
        self.assertEqual(parsed_record.weight_bytes, 256)
        self.assertEqual(parsed_record.operations, 8192)
        self.assertGreaterEqual(parsed_header.total_elapsed_ns, parsed_record.elapsed_ns)

    def test_bcuda_execute_graph_rejects_unsupported_layer_op(self):
        A = alloc_buffer(self.lib, 256, tensor_desc(BC_DTYPE_INT8, 2, 16))
        B = alloc_buffer(self.lib, 256, tensor_desc(BC_DTYPE_INT8, 2, 16))
        C = alloc_buffer(self.lib, 1024, tensor_desc(BC_DTYPE_INT32, 2, 16), usage=BC_USAGE_OUTPUT)
        descriptor = build_single_gemm_model_descriptor(A, B, C, op_type=BC_OP_NMS)
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        desc = BcudaGraphExecDesc(model, A, C, 0, 0)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(desc)), BC_STATUS_INVALID_COMMAND)

    def test_bcuda_execute_graph_runs_int8_relu_activation(self):
        inp = alloc_buffer(self.lib, 4, tensor_desc(BC_DTYPE_INT8, 1, 4))
        out = alloc_buffer(self.lib, 4, tensor_desc(BC_DTYPE_INT8, 1, 4), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, inp, bytes([253, 0, 5, 128]))
        descriptor = pack_model_descriptor(
            [
                (inp, BC_DTYPE_INT8, 1, 2, 4, 1, 4, 1),
                (out, BC_DTYPE_INT8, 1, 4, 4, 1, 4, 1),
            ],
            [layer_entry(BC_OP_ACTIVATION, input0=0, output=1)],
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, inp, out, 0, 0))), BC_STATUS_OK)
        self.assertEqual(read_buffer(self.lib, out, 4), bytes([0, 0, 5, 0]))

    def test_bcuda_execute_graph_runs_int32_relu_activation(self):
        inp = alloc_buffer(self.lib, 12, tensor_desc(BC_DTYPE_INT32, 1, 3))
        out = alloc_buffer(self.lib, 12, tensor_desc(BC_DTYPE_INT32, 1, 3), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, inp, int32_bytes([-7, 2, -1]))
        descriptor = pack_model_descriptor(
            [
                (inp, BC_DTYPE_INT32, 1, 2, 12, 1, 3, 1),
                (out, BC_DTYPE_INT32, 1, 4, 12, 1, 3, 1),
            ],
            [layer_entry(BC_OP_ACTIVATION, input0=0, output=1)],
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, inp, out, 0, 0))), BC_STATUS_OK)
        self.assertEqual(read_buffer(self.lib, out, 12), int32_bytes([0, 2, 0]))

    def test_bcuda_execute_graph_requantizes_int32_to_int8(self):
        inp = alloc_buffer(self.lib, 20, tensor_desc(BC_DTYPE_INT32, 1, 5))
        scale = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT32, 1, 4))
        out = alloc_buffer(self.lib, 5, tensor_desc(BC_DTYPE_INT8, 1, 5), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, inp, int32_bytes([-300, -2, 0, 8, 500]))
        write_buffer(self.lib, scale, int32_bytes([1, 1, 0, 3]))
        descriptor = pack_model_descriptor(
            [
                (inp, BC_DTYPE_INT32, 1, 2, 20, 1, 5, 1),
                (scale, BC_DTYPE_INT32, 1, 1, 16, 1, 4, 1),
                (out, BC_DTYPE_INT8, 1, 4, 5, 1, 5, 1),
            ],
            [layer_entry(BC_OP_REQUANTIZE, input0=0, output=2, scale=1)],
            output_index=2,
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, inp, out, 0, 0))), BC_STATUS_OK)
        self.assertEqual(read_buffer(self.lib, out, 5), bytes([128, 2, 3, 7, 127]))

    def test_bcuda_execute_graph_requantizes_to_tiled_transformer_activation(self):
        inp = alloc_buffer(self.lib, 24, tensor_desc(BC_DTYPE_INT32, 2, 3, 2))
        scale = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT32, 1, 4))
        out = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT8, 2, 3, 2, layout=BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, inp, int32_bytes([1, 2, 3, 4, 5, 6]))
        write_buffer(self.lib, scale, int32_bytes([1, 0, 0, 0]))
        descriptor = pack_model_descriptor(
            [
                (inp, BC_DTYPE_INT32, BC_LAYOUT_ROW_MAJOR, 2, 24, 2, 3, 2),
                (scale, BC_DTYPE_INT32, BC_LAYOUT_ROW_MAJOR, 1, 16, 1, 4, 1),
                (out, BC_DTYPE_INT8, BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8, 4, 16, 2, 3, 2),
            ],
            [layer_entry(BC_OP_REQUANTIZE, input0=0, output=2, scale=1, n=3, c=2)],
            output_index=2,
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, inp, out, 0, 0))), BC_STATUS_OK)
        self.assertEqual(read_buffer(self.lib, out, 16), bytes([1, 3, 5, 0, 0, 0, 0, 0, 2, 4, 6, 0, 0, 0, 0, 0]))

    def test_bcuda_execute_graph_transposes_activation_into_dynamic_b_layout(self):
        inp = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT8, 2, 3, 2, layout=BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8))
        out = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT8, 2, 2, 3, layout=BC_LAYOUT_TRANSFORMER_DYNAMIC_B_TILED_INT8), usage=BC_USAGE_OUTPUT)
        packed = bytes([1, 3, 5, 0, 0, 0, 0, 0, 2, 4, 6, 0, 0, 0, 0, 0])
        write_buffer(self.lib, inp, packed)
        descriptor = pack_model_descriptor(
            [
                (inp, BC_DTYPE_INT8, BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8, 2, 16, 2, 3, 2),
                (out, BC_DTYPE_INT8, BC_LAYOUT_TRANSFORMER_DYNAMIC_B_TILED_INT8, 4, 16, 2, 2, 3),
            ],
            [layer_entry(BC_OP_ACTIVATION_B_PACK, input0=0, output=1, n=2, c=3, flags=1)],
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, inp, out, 0, 0))), BC_STATUS_OK)
        self.assertEqual(read_buffer(self.lib, out, 16), packed)

    def test_bcuda_execute_graph_runs_reshape_copy(self):
        inp = alloc_buffer(self.lib, 6, tensor_desc(BC_DTYPE_INT8, 2, 2))
        out = alloc_buffer(self.lib, 6, tensor_desc(BC_DTYPE_INT8, 2, 3), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, inp, bytes([1, 2, 3, 4, 5, 6]))
        descriptor = pack_model_descriptor(
            [
                (inp, BC_DTYPE_INT8, 1, 2, 6, 2, 2, 3),
                (out, BC_DTYPE_INT8, 1, 4, 6, 2, 3, 2),
            ],
            [layer_entry(BC_OP_RESHAPE, input0=0, output=1)],
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, inp, out, 0, 0))), BC_STATUS_OK)
        self.assertEqual(read_buffer(self.lib, out, 6), bytes([1, 2, 3, 4, 5, 6]))

    def test_bcuda_execute_graph_runs_int8_maxpool_nchw(self):
        inp = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT8, 4, 1))
        out = alloc_buffer(self.lib, 4, tensor_desc(BC_DTYPE_INT8, 4, 1), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, inp, bytes([1, 2, 3, 4, 5, 250, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
        descriptor = pack_model_descriptor(
            [
                (inp, BC_DTYPE_INT8, 2, 2, 16, 4, 1, 1),
                (out, BC_DTYPE_INT8, 2, 4, 4, 4, 1, 1),
            ],
            [layer_entry(BC_OP_POOL, input0=0, output=1, n=1, c=1, h=4, w=4, out_h=2, out_w=2, kernel_h=2, kernel_w=2, stride_h=2, stride_w=2)],
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, inp, out, 0, 0))), BC_STATUS_OK)
        self.assertEqual(read_buffer(self.lib, out, 4), bytes([5, 8, 14, 16]))

    def test_bcuda_execute_graph_runs_int8_average_and_global_pool_nchw(self):
        inp = alloc_buffer(self.lib, 9, tensor_desc(BC_DTYPE_INT8, 4, 1))
        pooled = alloc_buffer(self.lib, 4, tensor_desc(BC_DTYPE_INT8, 4, 1))
        out = alloc_buffer(self.lib, 1, tensor_desc(BC_DTYPE_INT8, 4, 1), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, inp, bytes([249, 254, 3, 8, 4, 251, 6, 9, 255]))
        descriptor = pack_model_descriptor(
            [(inp, BC_DTYPE_INT8, 2, 2, 9, 3, 1, 1), (pooled, BC_DTYPE_INT8, 2, 3, 4, 2, 1, 1), (out, BC_DTYPE_INT8, 2, 4, 1, 1, 1, 1)],
            [
                layer_entry(BC_OP_POOL, input0=0, output=1, n=1, c=1, h=3, w=3, out_h=2, out_w=2, kernel_h=2, kernel_w=2, stride_h=1, stride_w=1, flags=1),
                layer_entry(BC_OP_POOL, input0=0, output=2, n=1, c=1, h=3, w=3, out_h=1, out_w=1, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, flags=1),
            ],
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, inp, out, 0, 0))), BC_STATUS_OK)
        self.assertEqual(read_buffer(self.lib, pooled, 4), bytes([0, 0, 6, 1]))
        self.assertEqual(read_buffer(self.lib, out, 1), bytes([1]))

    def test_bcuda_execute_graph_runs_conv2d_int8_to_int32(self):
        inp = alloc_buffer(self.lib, 9, tensor_desc(BC_DTYPE_INT8, 4, 1))
        weight = alloc_buffer(self.lib, 4, tensor_desc(BC_DTYPE_INT8, 4, 1))
        bias = alloc_buffer(self.lib, 4, tensor_desc(BC_DTYPE_INT32, 1, 1))
        out = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT32, 4, 1), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, inp, bytes([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        write_buffer(self.lib, weight, bytes([1, 0, 0, 1]))
        write_buffer(self.lib, bias, int32_bytes([1]))
        descriptor = pack_model_descriptor(
            [
                (inp, BC_DTYPE_INT8, 2, 2, 9, 4, 1, 1),
                (weight, BC_DTYPE_INT8, 1, 1, 4, 4, 1, 1),
                (bias, BC_DTYPE_INT32, 1, 1, 4, 1, 1, 1),
                (out, BC_DTYPE_INT32, 2, 4, 16, 4, 1, 1),
            ],
            [
                layer_entry(
                    BC_OP_CONV2D_GEMM,
                    input0=0,
                    output=3,
                    weight=1,
                    bias=2,
                    n=1,
                    c=1,
                    h=3,
                    w=3,
                    out_c=1,
                    out_h=2,
                    out_w=2,
                    kernel_h=2,
                    kernel_w=2,
                    stride_h=1,
                    stride_w=1,
                )
            ],
            output_index=3,
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, inp, out, 0, 0))), BC_STATUS_OK)
        self.assertEqual(read_buffer(self.lib, out, 16), int32_bytes([7, 9, 13, 15]))

    def test_bcuda_execute_graph_runs_presliced_conv2d_int8_to_int32(self):
        inp = alloc_buffer(self.lib, 9, tensor_desc(BC_DTYPE_INT8, 4, 1))
        weight = alloc_buffer(self.lib, 256, tensor_desc(BC_DTYPE_INT8, 4, 1))
        out = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT32, 4, 1), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, inp, bytes([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        packed = bytearray(256)
        packed[0], packed[16], packed[32], packed[48] = 1, 0, 0, 1
        write_buffer(self.lib, weight, bytes(packed))
        descriptor = pack_model_descriptor(
            [
                (inp, BC_DTYPE_INT8, 2, 2, 9, 4, 1, 1),
                (weight, BC_DTYPE_INT8, 4, 1, 256, 4, 1, 1),
                (out, BC_DTYPE_INT32, 2, 4, 16, 4, 1, 1),
            ],
            [layer_entry(BC_OP_CONV2D_GEMM, input0=0, output=2, weight=1, n=1, c=1, h=3, w=3, out_c=1, out_h=2, out_w=2, kernel_h=2, kernel_w=2, stride_h=1, stride_w=1, flags=1)],
            output_index=2,
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, inp, out, 0, 0))), BC_STATUS_OK)
        self.assertEqual(read_buffer(self.lib, out, 16), int32_bytes([6, 8, 12, 14]))

    def test_bcuda_execute_graph_runs_nearest_upsample_nchw(self):
        inp = alloc_buffer(self.lib, 4, tensor_desc(BC_DTYPE_INT8, 4, 1))
        out = alloc_buffer(self.lib, 16, tensor_desc(BC_DTYPE_INT8, 4, 1), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, inp, bytes([1, 2, 3, 4]))
        descriptor = pack_model_descriptor(
            [
                (inp, BC_DTYPE_INT8, 2, 2, 4, 4, 1, 1),
                (out, BC_DTYPE_INT8, 2, 4, 16, 4, 1, 1),
            ],
            [layer_entry(BC_OP_UPSAMPLE, input0=0, output=1, n=1, c=1, h=2, w=2, out_h=4, out_w=4)],
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, inp, out, 0, 0))), BC_STATUS_OK)
        self.assertEqual(read_buffer(self.lib, out, 16), bytes([1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]))

    def test_bcuda_execute_graph_runs_channel_concat_nchw(self):
        a = alloc_buffer(self.lib, 4, tensor_desc(BC_DTYPE_INT8, 4, 1))
        b = alloc_buffer(self.lib, 4, tensor_desc(BC_DTYPE_INT8, 4, 1))
        out = alloc_buffer(self.lib, 8, tensor_desc(BC_DTYPE_INT8, 4, 1), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, a, bytes([1, 2, 3, 4]))
        write_buffer(self.lib, b, bytes([5, 6, 7, 8]))
        descriptor = pack_model_descriptor(
            [
                (a, BC_DTYPE_INT8, 2, 2, 4, 4, 1, 1),
                (b, BC_DTYPE_INT8, 2, 2, 4, 4, 1, 1),
                (out, BC_DTYPE_INT8, 2, 4, 8, 4, 1, 2),
            ],
            [layer_entry(BC_OP_CONCAT, input0=0, input1=1, output=2, n=1, h=2, w=2, out_c=2)],
            output_index=2,
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, a, out, 0, 0))), BC_STATUS_OK)
        self.assertEqual(read_buffer(self.lib, out, 8), bytes([1, 2, 3, 4, 5, 6, 7, 8]))

    def _run_transformer_layer(self, buffers, layer, input_id, output_id):
        entries = [(buffer_id, dtype, 1, usage, size, 1, size, 1) for buffer_id, dtype, usage, size in buffers]
        descriptor = pack_model_descriptor(entries, [layer], output_index=next(i for i, item in enumerate(buffers) if item[0] == output_id))
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        status = self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, input_id, output_id, 0, 0)))
        self.assertEqual(status, BC_STATUS_OK)

    def test_bcuda_execute_graph_runs_transformer_elementwise_ops(self):
        inp = alloc_buffer(self.lib, 7, tensor_desc(BC_DTYPE_INT8, 1, 7))
        out = alloc_buffer(self.lib, 7, tensor_desc(BC_DTYPE_INT8, 1, 7), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, inp, bytes([224, 232, 248, 0, 8, 24, 32]))
        self._run_transformer_layer([(inp, BC_DTYPE_INT8, 2, 7), (out, BC_DTYPE_INT8, 4, 7)], layer_entry(BC_OP_GELU), inp, out)
        self.assertEqual(read_buffer(self.lib, out, 7), bytes([0, 0, 253, 0, 5, 24, 32]))

        rhs = alloc_buffer(self.lib, 7, tensor_desc(BC_DTYPE_INT8, 1, 7))
        write_buffer(self.lib, rhs, bytes([127] * 7))
        self._run_transformer_layer([(inp, BC_DTYPE_INT8, 2, 7), (rhs, BC_DTYPE_INT8, 2, 7), (out, BC_DTYPE_INT8, 4, 7)], layer_entry(BC_OP_RESIDUAL_ADD, input1=1, output=2), inp, out)
        self.assertEqual(read_buffer(self.lib, out, 7)[-1], 127)

    def test_bcuda_execute_graph_concatenates_token_heads(self):
        a = alloc_buffer(self.lib, 4, tensor_desc(BC_DTYPE_INT8, 2, 2, 2))
        b = alloc_buffer(self.lib, 2, tensor_desc(BC_DTYPE_INT8, 2, 2, 1))
        out = alloc_buffer(self.lib, 6, tensor_desc(BC_DTYPE_INT8, 2, 2, 3), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, a, bytes([1, 2, 3, 4]))
        write_buffer(self.lib, b, bytes([5, 6]))
        descriptor = pack_model_descriptor(
            [
                (a, BC_DTYPE_INT8, BC_LAYOUT_ROW_MAJOR, 2, 4, 2, 2, 2),
                (b, BC_DTYPE_INT8, BC_LAYOUT_ROW_MAJOR, 2, 2, 2, 2, 1),
                (out, BC_DTYPE_INT8, BC_LAYOUT_ROW_MAJOR, 4, 6, 2, 2, 3),
            ],
            [layer_entry(BC_OP_TOKEN_CONCAT, input0=0, input1=1, input2=0, output=2, n=2, c=2, h=1, w=0, out_c=3)],
            output_index=2,
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, a, out, 0, 0))), BC_STATUS_OK)
        self.assertEqual(read_buffer(self.lib, out, 6), bytes([1, 2, 5, 3, 4, 6]))

    def test_bcuda_execute_graph_builds_class_and_position_tokens(self):
        patches = alloc_buffer(self.lib, 4, tensor_desc(BC_DTYPE_INT8, 2, 2, 2))
        position = alloc_buffer(self.lib, 6, tensor_desc(BC_DTYPE_INT8, 2, 3, 2))
        scale = alloc_buffer(self.lib, 256, tensor_desc(BC_DTYPE_INT8, 1, 256))
        out = alloc_buffer(self.lib, 6, tensor_desc(BC_DTYPE_INT8, 2, 3, 2), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, patches, bytes([5, 251, 7, 249]))
        write_buffer(self.lib, position, bytes([10, 11, 1, 2, 3, 4]))
        write_buffer(self.lib, scale, bytes(range(256)))
        descriptor = pack_model_descriptor(
            [
                (patches, BC_DTYPE_INT8, BC_LAYOUT_ROW_MAJOR, 2, 4, 2, 2, 2),
                (position, BC_DTYPE_INT8, BC_LAYOUT_ROW_MAJOR, 1, 6, 2, 3, 2),
                (scale, BC_DTYPE_INT8, BC_LAYOUT_ROW_MAJOR, 1, 256, 1, 256, 1),
                (out, BC_DTYPE_INT8, BC_LAYOUT_ROW_MAJOR, 4, 6, 2, 3, 2),
            ],
            [layer_entry(BC_OP_TOKEN_EMBEDDING, input0=0, weight=1, scale=2,
                         output=3, n=2, c=2)],
            output_index=3,
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        self.assertEqual(
            self.lib.bcuda_execute_graph(
                ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, patches, out, 0, 0))),
            BC_STATUS_OK,
        )
        self.assertEqual(read_buffer(self.lib, out, 6), bytes([10, 11, 6, 253, 10, 253]))

    def test_bcuda_execute_graph_runs_layernorm_and_softmax(self):
        inp = alloc_buffer(self.lib, 4, tensor_desc(BC_DTYPE_INT8, 1, 4))
        gamma = alloc_buffer(self.lib, 8, tensor_desc(BC_DTYPE_INT16, 1, 4))
        beta = alloc_buffer(self.lib, 8, tensor_desc(BC_DTYPE_INT16, 1, 4))
        out = alloc_buffer(self.lib, 4, tensor_desc(BC_DTYPE_INT8, 1, 4), usage=BC_USAGE_OUTPUT)
        write_buffer(self.lib, inp, bytes([253, 255, 1, 3]))
        write_buffer(self.lib, gamma, b"".join((256).to_bytes(2, "little", signed=True) for _ in range(4)))
        write_buffer(self.lib, beta, bytes(8))
        buffers = [(inp, BC_DTYPE_INT8, 2, 4), (gamma, BC_DTYPE_INT16, 1, 8), (beta, BC_DTYPE_INT16, 1, 8), (out, BC_DTYPE_INT8, 4, 4)]
        self._run_transformer_layer(buffers, layer_entry(BC_OP_LAYERNORM, output=3, scale=1, bias=2, n=1, c=4), inp, out)
        self.assertEqual(read_buffer(self.lib, out, 4), bytes([254, 255, 1, 2]))

        tiled_out = alloc_buffer(self.lib, 32, tensor_desc(BC_DTYPE_INT8, 2, 1, 4, layout=BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8), usage=BC_USAGE_OUTPUT)
        descriptor = pack_model_descriptor(
            [
                (inp, BC_DTYPE_INT8, BC_LAYOUT_ROW_MAJOR, 2, 4, 2, 1, 4),
                (gamma, BC_DTYPE_INT16, BC_LAYOUT_ROW_MAJOR, 1, 8, 1, 4, 1),
                (beta, BC_DTYPE_INT16, BC_LAYOUT_ROW_MAJOR, 1, 8, 1, 4, 1),
                (tiled_out, BC_DTYPE_INT8, BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8, 4, 32, 2, 1, 4),
            ],
            [layer_entry(BC_OP_LAYERNORM, output=3, scale=1, bias=2, n=1, c=4)],
            output_index=3,
        )
        model = alloc_buffer(self.lib, len(descriptor), tensor_desc(BC_DTYPE_INT8, 1, len(descriptor)), usage=BC_USAGE_DESCRIPTOR)
        write_buffer(self.lib, model, descriptor)
        write_buffer(self.lib, inp, bytes([253, 255, 1, 3]))
        self.assertEqual(self.lib.bcuda_execute_graph(ctypes.byref(self.runtime), ctypes.byref(BcudaGraphExecDesc(model, inp, tiled_out, 0, 0))), BC_STATUS_OK)
        expected_tiled = bytearray(32)
        expected_tiled[0], expected_tiled[8], expected_tiled[16], expected_tiled[24] = 254, 255, 1, 2
        self.assertEqual(read_buffer(self.lib, tiled_out, 32), bytes(expected_tiled))

        write_buffer(self.lib, inp, bytes([5, 5, 5, 5]))
        self._run_transformer_layer([(inp, BC_DTYPE_INT8, 2, 4), (out, BC_DTYPE_INT8, 4, 4)], layer_entry(BC_OP_SOFTMAX, n=1, c=4), inp, out)
        self.assertEqual(read_buffer(self.lib, out, 4), bytes([32, 32, 32, 32]))


if __name__ == "__main__":
    unittest.main()
