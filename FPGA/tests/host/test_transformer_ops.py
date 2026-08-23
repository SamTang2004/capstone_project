import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda import Device, DiagnosticResult, DeiTConfig, build_deit_attention_block_package, build_deit_mlp_block_package
from budgetcuda.compiler import GraphBuilder, HostBuffer, compose_model_packages, upload_model
from budgetcuda.dtypes import BufferUsage, DType, Layout, OpType
from budgetcuda.diagnostics import (
    DIAGNOSTICS_HEADER_STRUCT,
    DIAGNOSTICS_RECORD_STRUCT,
    DiagnosticsSettings,
    parse_diagnostics,
)
from budgetcuda.packing import (
    pack_transformer_activations_int8,
    pack_transformer_weights,
    unpack_transformer_activations_int8,
    unpack_transformer_weights,
)


def values(result):
    return result.reshape(-1).tolist() if hasattr(result, "reshape") else list(result)


class TransformerOpsTest(unittest.TestCase):
    def test_transformer_layout_abi_and_tile_packing(self):
        self.assertEqual([int(Layout.TRANSFORMER_ACTIVATION_TILED_INT8),
                          int(Layout.TRANSFORMER_WEIGHT_TILED_INT8),
                          int(Layout.TRANSFORMER_WEIGHT_TILED_INT4)], [5, 6, 7])
        activation = pack_transformer_activations_int8([[1, 2], [3, 4], [5, 6]], 3, 2)
        self.assertEqual(list(activation[:8]), [1, 3, 5, 0, 0, 0, 0, 0])
        self.assertEqual(list(activation[8:16]), [2, 4, 6, 0, 0, 0, 0, 0])
        w4 = pack_transformer_weights([[-8, -1, 0, 7], [1, 2, 3, 4]], 2, 4, int4=True)
        self.assertEqual(list(w4), [0xF8, 0x70, 0, 0, 0x21, 0x43, 0, 0])
        self.assertEqual(HostBuffer("a", (3, 2), DType.INT8, Layout.TRANSFORMER_ACTIVATION_TILED_INT8).size_bytes, 16)
        self.assertEqual(HostBuffer("w8", (2, 9), DType.INT8, Layout.TRANSFORMER_WEIGHT_TILED_INT8).size_bytes, 32)
        self.assertEqual(HostBuffer("w4", (2, 9), DType.INT8, Layout.TRANSFORMER_WEIGHT_TILED_INT4).size_bytes, 16)

    def test_packed_weight_round_trip(self):
        values_in = [[-8, -1, 0, 7, 2], [1, 2, 3, 4, -5]]
        for int4 in (False, True):
            raw = pack_transformer_weights(values_in, 2, 5, int4=int4)
            self.assertEqual(unpack_transformer_weights(raw, 2, 5, int4=int4), values_in)

    def test_packed_activation_round_trip_with_padding(self):
        values_in = [[row * 7 + column - 20 for column in range(7)] for row in range(11)]
        raw = pack_transformer_activations_int8(values_in, 11, 7)
        self.assertEqual(unpack_transformer_activations_int8(raw, 11, 7), values_in)

    def test_opcodes_append_without_changing_legacy_values(self):
        self.assertEqual(int(OpType.REQUANTIZE), 11)
        self.assertEqual([int(OpType.LAYERNORM), int(OpType.GELU), int(OpType.SOFTMAX), int(OpType.RESIDUAL_ADD)], [12, 13, 14, 15])
        self.assertEqual(int(OpType.ACTIVATION_B_PACK), 16)
        self.assertEqual(int(OpType.TOKEN_EMBEDDING), 18)

    def test_token_embedding_prepends_class_position_and_adds_patches(self):
        graph = GraphBuilder("tokens")
        patches = graph.add_buffer("patches", (2, 2), DType.INT8)
        position = graph.add_buffer(
            "position", (3, 2), DType.INT8, BufferUsage.WEIGHT,
            [[10, 11], [1, 2], [3, 4]])
        identity_lut = list(range(128)) + list(range(-128, 0))
        scale = graph.add_buffer(
            "scale", (256,), DType.INT8, BufferUsage.WEIGHT, identity_lut)
        output = graph.add_buffer("output", (3, 2), DType.INT8, BufferUsage.OUTPUT)
        graph.token_embedding(patches, position, scale, output, patch_rows=2, features=2)
        model = upload_model(Device.mock(), graph.package(patches, output))
        self.assertEqual(values(model.run([[5, -5], [7, -7]])), [10, 11, 6, -3, 10, -3])

    def test_sequential_packages_compose_to_one_graph(self):
        first = GraphBuilder("first")
        first_input = first.add_buffer("input", (4,), DType.INT8)
        first_output = first.add_buffer("output", (4,), DType.INT8, BufferUsage.OUTPUT)
        first.relu(first_input, first_output)
        second = GraphBuilder("second")
        second_input = second.add_buffer("input", (4,), DType.INT8)
        second_output = second.add_buffer("output", (4,), DType.INT8, BufferUsage.OUTPUT)
        second.relu(second_input, second_output)
        package = compose_model_packages(
            "composed", [first.package(first_input, first_output),
                         second.package(second_input, second_output)])
        self.assertEqual((len(package.buffers), len(package.layers)), (3, 2))
        model = upload_model(Device.mock(), package)
        self.assertEqual(values(model.run([-4, 0, 3, 8])), [0, 0, 3, 8])

    def test_layernorm_matches_rtl_golden_vector(self):
        graph = GraphBuilder("ln")
        inp = graph.add_buffer("input", (1, 4), DType.INT8)
        gamma = graph.add_buffer("gamma", (4,), DType.INT16, BufferUsage.WEIGHT, [256] * 4)
        beta = graph.add_buffer("beta", (4,), DType.INT16, BufferUsage.WEIGHT, [0] * 4)
        out = graph.add_buffer("output", (1, 4), DType.INT8, BufferUsage.OUTPUT)
        graph.layer_norm(inp, gamma, beta, out, rows=1, features=4)
        model = upload_model(Device.mock(), graph.package(inp, out))
        self.assertEqual(values(model.run([-3, -1, 1, 3])), [-2, -1, 1, 2])

    def test_gelu_matches_piecewise_hardware_approximation(self):
        graph = GraphBuilder("gelu")
        inp = graph.add_buffer("input", (7,), DType.INT8)
        out = graph.add_buffer("output", (7,), DType.INT8, BufferUsage.OUTPUT)
        graph.gelu(inp, out)
        model = upload_model(Device.mock(), graph.package(inp, out))
        self.assertEqual(values(model.run([-32, -24, -8, 0, 8, 24, 32])), [0, 0, -3, 0, 5, 24, 32])

    def test_softmax_q7_and_saturating_residual(self):
        graph = GraphBuilder("softmax")
        inp = graph.add_buffer("input", (1, 4), DType.INT8)
        probs = graph.add_buffer("probs", (1, 4), DType.INT8, BufferUsage.OUTPUT)
        graph.softmax(inp, probs, rows=1, length=4)
        model = upload_model(Device.mock(), graph.package(inp, probs))
        self.assertEqual(values(model.run([5, 5, 5, 5])), [32, 32, 32, 32])

        graph = GraphBuilder("residual")
        lhs = graph.add_buffer("lhs", (4,), DType.INT8)
        rhs = graph.add_buffer("rhs", (4,), DType.INT8, data=[20, -20, 100, -100])
        out = graph.add_buffer("output", (4,), DType.INT8, BufferUsage.OUTPUT)
        graph.residual_add(lhs, rhs, out)
        model = upload_model(Device.mock(), graph.package(lhs, out))
        self.assertEqual(values(model.run([120, -120, 50, -50])), [127, -128, 127, -128])

    def test_transformer_diagnostics_preserve_v3_record_abi(self):
        graph = GraphBuilder("debug")
        inp = graph.add_buffer("input", (4,), DType.INT8)
        out = graph.add_buffer("output", (4,), DType.INT8, BufferUsage.OUTPUT)
        graph.gelu(inp, out)
        model = upload_model(Device.mock(), graph.package(inp, out))
        result = model.run([-8, 0, 8, 24], diagnostics=True)
        self.assertIsInstance(result, DiagnosticResult)
        self.assertEqual(values(result.output), [-3, 0, 5, 24])
        self.assertEqual(result.diagnostics.layers[0].op_name, "GELU")
        self.assertEqual(result.diagnostics.layers[0].output_bytes, 4)
        self.assertEqual(result.diagnostics.layers[0].transformer_cycles["memory_stall_cycles"], 0)

    def test_v3_diagnostics_preserve_counters_beyond_32_bits(self):
        settings = DiagnosticsSettings()
        self.assertEqual(settings.peak_operations_per_cycle, 256)
        raw = bytearray(settings.initial_buffer(1))
        large = (1 << 40) + 123
        header = list(DIAGNOSTICS_HEADER_STRUCT.unpack_from(raw))
        header[4] = 1
        DIAGNOSTICS_HEADER_STRUCT.pack_into(raw, 0, *header)
        record = (0, int(OpType.GEMM), 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                  large, large - 1, large - 2, large - 3, large - 4,
                  large - 5, large - 6, 7, 0)
        DIAGNOSTICS_RECORD_STRUCT.pack_into(
            raw, DIAGNOSTICS_HEADER_STRUCT.size, *record)
        parsed = parse_diagnostics(raw, settings).layers[0].transformer_cycles
        self.assertEqual(parsed["window_cycles"], large)
        self.assertEqual(parsed["max_op_cycles"], large - 6)
        self.assertEqual(parsed["completed_ops"], 7)

    def test_deit_mlp_builder_runs_pre_norm_residual_graph(self):
        self.assertEqual(DeiTConfig().tokens, 197)
        package = build_deit_mlp_block_package(1, 4, 8, [0] * 32, [0] * 32)
        self.assertEqual([layer.op_type for layer in package.layers], [
            OpType.LAYERNORM, OpType.GEMM, OpType.REQUANTIZE, OpType.GELU,
            OpType.GEMM, OpType.REQUANTIZE, OpType.RESIDUAL_ADD,
        ])
        model = upload_model(Device.mock(), package)
        self.assertEqual(values(model.run([-3, -1, 1, 3])), [-3, -1, 1, 3])

    def test_deit_mlp_accelerator_layout_runs_w8_and_w4(self):
        for int4 in (False, True):
            package = build_deit_mlp_block_package(
                3, 4, 8, [0] * 32, [0] * 32,
                accelerator_layout=True, weight_int4=int4,
            )
            expected_weight_layout = (Layout.TRANSFORMER_WEIGHT_TILED_INT4 if int4
                                      else Layout.TRANSFORMER_WEIGHT_TILED_INT8)
            self.assertEqual(package.buffers[package.name_to_index["deit_mlp.norm.output"]].layout,
                             Layout.TRANSFORMER_ACTIVATION_TILED_INT8)
            self.assertEqual(package.buffers[package.name_to_index["deit_mlp.fc1.weight"]].layout,
                             expected_weight_layout)
            self.assertEqual(package.buffers[package.name_to_index["deit_mlp.fc1.q"]].size_bytes, 64)
            model = upload_model(Device.mock(), package)
            sample = [-3, -1, 1, 3] * 3
            self.assertEqual(values(model.run(sample)), sample)

    def test_transformer_gemm_applies_exported_int32_bias(self):
        graph = GraphBuilder("biased")
        inp = graph.add_transformer_activation("input", 1, 2)
        weight = graph.add_transformer_weight("weight", 2, 2, [[0, 0], [0, 0]])
        bias = graph.add_buffer("bias", (2,), DType.INT32, BufferUsage.WEIGHT, [3, -4])
        acc = graph.add_buffer("acc", (1, 2), DType.INT32, BufferUsage.OUTPUT)
        graph.transformer_gemm(inp, weight, acc, m=1, n=2, k=2, bias_idx=bias)
        model = upload_model(Device.mock(), graph.package(inp, acc))
        packed = pack_transformer_activations_int8([[1, 2]], 1, 2)
        self.assertEqual(values(model.run(packed)), [3, -4])

    def test_dynamic_transpose_pack_enables_qk_transpose_gemm(self):
        graph = GraphBuilder("attention")
        inp = graph.add_buffer("input", (3, 2), DType.INT8)
        gamma = graph.add_buffer("gamma", (2,), DType.INT16, BufferUsage.WEIGHT, [256, 256])
        beta = graph.add_buffer("beta", (2,), DType.INT16, BufferUsage.WEIGHT, [0, 0])
        qk = graph.add_transformer_activation("qk", 3, 2)
        dynamic_b = graph.add_transformer_dynamic_b("k_transposed", 2, 3)
        scores = graph.add_buffer("scores", (3, 3), DType.INT32, BufferUsage.OUTPUT)
        graph.layer_norm(inp, gamma, beta, qk, rows=3, features=2)
        graph.activation_b_pack(qk, dynamic_b, reduction=2, outputs=3, transpose=True)
        graph.transformer_gemm(qk, dynamic_b, scores, m=3, n=3, k=2)
        package = graph.package(inp, scores)
        self.assertEqual(package.buffers[dynamic_b].layout, Layout.TRANSFORMER_DYNAMIC_B_TILED_INT8)
        self.assertEqual(package.buffers[dynamic_b].size_bytes, 16)
        model = upload_model(Device.mock(), package)
        self.assertEqual(values(model.run([-3, 3, -1, 1, 2, -2])), [5, 5, -4, 5, 5, -4, -4, -4, 5])

    def test_complete_three_head_attention_schedule_preserves_zero_projection_residual(self):
        weights = [0] * 36
        package = build_deit_attention_block_package(
            2, 6, 3, weights, weights, weights, weights, weight_int4=True,
        )
        op_types = [layer.op_type for layer in package.layers]
        self.assertEqual(op_types.count(OpType.GEMM), 16)
        self.assertEqual(op_types.count(OpType.SOFTMAX), 3)
        self.assertEqual(op_types.count(OpType.ACTIVATION_B_PACK), 10)
        self.assertEqual(op_types.count(OpType.TOKEN_CONCAT), 1)
        model = upload_model(Device.mock(), package)
        sample = [-3, -2, -1, 1, 2, 3] * 2
        self.assertEqual(values(model.run(sample)), sample)


if __name__ == "__main__":
    unittest.main()
