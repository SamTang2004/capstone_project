from __future__ import annotations

import struct
from time import perf_counter_ns
from .dtypes import Command, DType, Layout, OpType, Status
from .compiler import CONV_FLAG_PRESLICED_WEIGHTS
from .protocol import build_ok_response
from .serializer import (
    EXECUTE_REQUEST_TAIL_STRUCT,
    EXECUTE_RESPONSE_TAIL_STRUCT,
    FREE_REQUEST_TAIL_STRUCT,
    FREE_RESPONSE_TAIL_STRUCT,
    GRAPH_EXEC_DESC_STRUCT,
    HEADER_STRUCT,
    READ_REQUEST_TAIL_STRUCT,
    READ_RESPONSE_TAIL_STRUCT,
    RESPONSE_HEADER_STRUCT,
    TENSOR_DESC_STRUCT,
    WRITE_REQUEST_TAIL_STRUCT,
    WRITE_RESPONSE_TAIL_STRUCT,
    unpack_gemm_desc,
)

MODEL_DESC_STRUCT = struct.Struct("<IIIIIIIIIIIIII")
BUFFER_DESC_STRUCT = struct.Struct("<IIIIIIII")
LAYER_DESC_STRUCT = struct.Struct("<IIIIIIIIIIIIIIIIIIIIIIII")
MODEL_MAGIC = 0x444D4342
MODEL_VERSION = 1
DIAGNOSTICS_HEADER_STRUCT = struct.Struct("<IIIIIIIIQQ")
DIAGNOSTICS_RECORD_STRUCT = struct.Struct("<IIIIQQQQQQQII16I")


class MockBackend:
    def __init__(self):
        self._next_buffer_id = 1
        self.buffers: dict[int, bytearray] = {}

    def transact(self, request: bytes) -> bytes:
        header = HEADER_STRUCT.unpack_from(request)
        _, _, command, request_id, header_bytes, payload_bytes, _, _ = header
        payload = request[header_bytes : header_bytes + payload_bytes]

        if command == Command.ALLOC:
            return self._handle_alloc(request, request_id)
        if command == Command.WRITE:
            return self._handle_write(request, request_id, payload)
        if command == Command.READ:
            return self._handle_read(request, request_id)
        if command == Command.EXECUTE:
            return self._handle_execute(request, request_id, payload)
        if command == Command.FREE:
            return self._handle_free(request, request_id)
        return self._error_response(Command(command), request_id, Status.INVALID_COMMAND)

    def _handle_alloc(self, request: bytes, request_id: int) -> bytes:
        offset = HEADER_STRUCT.size
        size_bytes = int.from_bytes(request[offset : offset + 8], "little")
        buffer_id = self._next_buffer_id
        self._next_buffer_id += 1
        self.buffers[buffer_id] = bytearray(size_bytes)
        tail = struct.pack("<IIQ", buffer_id, 0, size_bytes)
        return build_ok_response(Command.ALLOC, request_id, tail)

    def _handle_write(self, request: bytes, request_id: int, payload: bytes) -> bytes:
        offset = HEADER_STRUCT.size
        buffer_id, _, offset_bytes, write_bytes = WRITE_REQUEST_TAIL_STRUCT.unpack_from(request, offset)
        buf = self._get_buffer(buffer_id)
        end = offset_bytes + write_bytes
        if end > len(buf):
            return self._error_response(Command.WRITE, request_id, Status.GENERAL_ERROR)
        buf[offset_bytes:end] = payload[:write_bytes]
        tail = WRITE_RESPONSE_TAIL_STRUCT.pack(buffer_id, 0, write_bytes)
        return build_ok_response(Command.WRITE, request_id, tail)

    def _handle_read(self, request: bytes, request_id: int) -> bytes:
        offset = HEADER_STRUCT.size
        buffer_id, _, offset_bytes, read_bytes = READ_REQUEST_TAIL_STRUCT.unpack_from(request, offset)
        buf = self._get_buffer(buffer_id)
        end = offset_bytes + read_bytes
        if end > len(buf):
            return self._error_response(Command.READ, request_id, Status.GENERAL_ERROR)
        payload = bytes(buf[offset_bytes:end])
        tail = READ_RESPONSE_TAIL_STRUCT.pack(buffer_id, 0, read_bytes)
        return build_ok_response(Command.READ, request_id, tail, payload)

    def _handle_execute(self, request: bytes, request_id: int, payload: bytes) -> bytes:
        offset = HEADER_STRUCT.size
        op_type, descriptor_bytes = EXECUTE_REQUEST_TAIL_STRUCT.unpack_from(request, offset)
        descriptor = payload[:descriptor_bytes]
        if op_type == OpType.GEMM:
            self._execute_gemm(descriptor)
        elif op_type == OpType.GRAPH:
            self._execute_graph(descriptor)
        else:
            return self._error_response(Command.EXECUTE, request_id, Status.INVALID_COMMAND)
        tail = EXECUTE_RESPONSE_TAIL_STRUCT.pack(op_type, Status.OK, 0)
        return build_ok_response(Command.EXECUTE, request_id, tail)

    def _handle_free(self, request: bytes, request_id: int) -> bytes:
        offset = HEADER_STRUCT.size
        buffer_id, _ = FREE_REQUEST_TAIL_STRUCT.unpack_from(request, offset)
        self._get_buffer(buffer_id)
        del self.buffers[buffer_id]
        tail = FREE_RESPONSE_TAIL_STRUCT.pack(buffer_id, 0)
        return build_ok_response(Command.FREE, request_id, tail)

    def _execute_gemm(self, descriptor: bytes) -> None:
        desc = unpack_gemm_desc(descriptor)
        if desc["A_dtype"] != DType.INT8 or desc["B_dtype"] != DType.INT8 or desc["C_dtype"] != DType.INT32:
            raise ValueError("Mock GEMM only supports INT8 x INT8 -> INT32")
        M, N, K = desc["M"], desc["N"], desc["K"]
        a_raw = self._get_buffer(desc["A_buffer"])
        b_raw = self._get_buffer(desc["B_buffer"])
        a_layout, b_layout = desc["A_layout"], desc["B_layout"]

        def read_a(m, k):
            if a_layout == Layout.TRANSFORMER_ACTIVATION_TILED_INT8:
                return self._signed8(a_raw[((m // 8) * K + k) * 8 + (m & 7)])
            return self._signed8(a_raw[m * K + k])

        def read_b(k, n):
            if b_layout in (Layout.TRANSFORMER_WEIGHT_TILED_INT8, Layout.TRANSFORMER_DYNAMIC_B_TILED_INT8):
                return self._signed8(b_raw[((n // 8) * K + k) * 8 + (n & 7)])
            if b_layout == Layout.TRANSFORMER_WEIGHT_TILED_INT4:
                packed = b_raw[((n // 8) * K + k) * 4 + ((n & 7) // 2)]
                nibble = (packed >> (4 if n & 1 else 0)) & 0xF
                return nibble if nibble < 8 else nibble - 16
            return self._signed8(b_raw[k * N + n])
        out = bytearray()
        for m in range(M):
            for n in range(N):
                acc = 0
                for k in range(K):
                    acc += read_a(m, k) * read_b(k, n)
                out += int(acc).to_bytes(4, "little", signed=True)
        self._get_buffer(desc["C_buffer"])[: len(out)] = out

    def _execute_graph(self, descriptor: bytes) -> None:
        model_buffer, input_buffer, output_buffer, diagnostics_flags, diagnostics_buffer = GRAPH_EXEC_DESC_STRUCT.unpack_from(descriptor)
        raw_model = self._get_buffer(model_buffer)
        model = MODEL_DESC_STRUCT.unpack_from(raw_model)
        magic, version, num_layers, layer_offset, num_buffers, buffer_offset, model_input, model_output, *_ = model
        if magic != MODEL_MAGIC or version != MODEL_VERSION:
            raise ValueError("Invalid Budget-CUDA model descriptor")
        table = [
            BUFFER_DESC_STRUCT.unpack_from(raw_model, buffer_offset + idx * BUFFER_DESC_STRUCT.size)
            for idx in range(num_buffers)
        ]
        if table[model_input][0] != input_buffer or table[model_output][0] != output_buffer:
            raise ValueError("Graph execute buffers do not match model descriptor")
        diag = None
        graph_started = perf_counter_ns()
        if diagnostics_flags or diagnostics_buffer:
            if not diagnostics_flags or not diagnostics_buffer:
                raise ValueError("Diagnostics flags and buffer must be provided together")
            diag = self._get_buffer(diagnostics_buffer)
            magic, version, flags, capacity, _, record_bytes, clock_hz, peak, _, _ = DIAGNOSTICS_HEADER_STRUCT.unpack_from(diag)
            if magic != 0x47444342 or version != 2 or flags != diagnostics_flags or capacity < num_layers or record_bytes != DIAGNOSTICS_RECORD_STRUCT.size:
                raise ValueError("Invalid diagnostics buffer")
        for idx in range(num_layers):
            layer = LAYER_DESC_STRUCT.unpack_from(raw_model, layer_offset + idx * LAYER_DESC_STRUCT.size)
            op_type = layer[0]
            layer_started = perf_counter_ns()
            if op_type == OpType.GEMM:
                self._execute_graph_gemm(table, layer)
            elif op_type == OpType.CONV2D_GEMM:
                self._execute_graph_conv2d(table, layer)
            elif op_type == OpType.ACTIVATION:
                self._execute_graph_relu(table, layer)
            elif op_type == OpType.REQUANTIZE:
                self._execute_graph_requantize(table, layer)
            elif op_type == OpType.POOL:
                self._execute_graph_pool(table, layer)
            elif op_type == OpType.UPSAMPLE:
                self._execute_graph_upsample(table, layer)
            elif op_type == OpType.CONCAT:
                self._execute_graph_concat(table, layer)
            elif op_type == OpType.RESHAPE:
                self._execute_graph_reshape(table, layer)
            elif op_type == OpType.LAYERNORM:
                self._execute_graph_layernorm(table, layer)
            elif op_type == OpType.GELU:
                self._execute_graph_gelu(table, layer)
            elif op_type == OpType.SOFTMAX:
                self._execute_graph_softmax(table, layer)
            elif op_type == OpType.RESIDUAL_ADD:
                self._execute_graph_residual(table, layer)
            elif op_type == OpType.ACTIVATION_B_PACK:
                self._execute_graph_activation_b_pack(table, layer)
            elif op_type == OpType.TOKEN_CONCAT:
                self._execute_graph_token_concat(table, layer)
            elif op_type == OpType.TOKEN_EMBEDDING:
                self._execute_graph_token_embedding(table, layer)
            else:
                raise ValueError(f"Unsupported graph op: {op_type}")
            if diag is not None:
                elapsed = max(1, perf_counter_ns() - layer_started)
                input_bytes, output_bytes = table[layer[1]][4], table[layer[4]][4]
                weight_bytes = table[layer[5]][4] if op_type == OpType.GEMM else 0
                if op_type == OpType.LAYERNORM: weight_bytes = table[layer[6]][4] + table[layer[7]][4]
                operations = 2 * layer[8] * layer[9] * layer[12] if op_type == OpType.GEMM else output_bytes
                cycles = elapsed * clock_hz // 1_000_000_000
                throughput = operations * 1_000_000_000 // elapsed
                utilization = min(1_000_000, operations * 1_000_000 // max(1, cycles * peak))
                record = (idx, int(op_type), 0, 0, elapsed, cycles, input_bytes, output_bytes, weight_bytes, operations, throughput, utilization, 0, *([0] * 16))
                DIAGNOSTICS_RECORD_STRUCT.pack_into(diag, DIAGNOSTICS_HEADER_STRUCT.size + idx * DIAGNOSTICS_RECORD_STRUCT.size, *record)
        if diag is not None:
            header = list(DIAGNOSTICS_HEADER_STRUCT.unpack_from(diag))
            header[4], header[8] = num_layers, perf_counter_ns() - graph_started
            DIAGNOSTICS_HEADER_STRUCT.pack_into(diag, 0, *header)

    def _execute_graph_gemm(self, table, layer) -> None:
        _, input0, _, _, output, weight, bias, _, n, c, _, _, out_c, *_ = layer
        desc = struct.pack(
            "<IIIIIIIIIIIIII",
            table[input0][0],
            table[weight][0],
            table[output][0],
            n,
            out_c,
            c,
            int(DType.INT8),
            int(DType.INT8),
            int(DType.INT32),
            table[input0][2],
            table[weight][2],
            table[output][2],
            layer[22],
            0,
        )
        self._execute_gemm(desc)
        if bias:
            output_raw = self._get_buffer(table[output][0])
            bias_raw = self._get_buffer(table[bias][0])
            if table[bias][1] != DType.INT32 or len(bias_raw) < out_c * 4:
                raise ValueError("Mock GEMM bias must contain N INT32 values")
            for row in range(n):
                for column in range(out_c):
                    offset = (row * out_c + column) * 4
                    value = int.from_bytes(output_raw[offset:offset + 4], "little", signed=True)
                    value += int.from_bytes(bias_raw[column * 4:column * 4 + 4], "little", signed=True)
                    output_raw[offset:offset + 4] = value.to_bytes(4, "little", signed=True)

    def _execute_graph_relu(self, table, layer) -> None:
        _, input0, _, _, output, *_, flags, _ = layer
        input_entry = table[input0]
        output_entry = table[output]
        src = self._get_buffer(input_entry[0])
        dst = self._get_buffer(output_entry[0])
        if input_entry[1] == DType.INT8:
            for idx, byte in enumerate(src[: input_entry[4]]):
                value = byte if byte < 128 else byte - 256
                dst[idx] = (value >> ((flags >> 8) & 7) if (flags & 1) and value < 0 else max(0, value)) & 0xFF
        elif input_entry[1] == DType.INT32:
            for idx in range(0, input_entry[4], 4):
                value = int.from_bytes(src[idx : idx + 4], "little", signed=True)
                value = value >> ((flags >> 8) & 7) if (flags & 1) and value < 0 else max(0, value)
                dst[idx : idx + 4] = value.to_bytes(4, "little", signed=True)
        else:
            raise ValueError("Mock ReLU supports INT8 and INT32")

    def _execute_graph_requantize(self, table, layer) -> None:
        _, input0, _, _, output, _, _, scale, rows, columns, *_ = layer
        input_entry = table[input0]
        output_entry = table[output]
        scale_entry = table[scale]
        if input_entry[1] != DType.INT32 or output_entry[1] != DType.INT8 or scale_entry[1] != DType.INT32:
            raise ValueError("Mock requantize requires INT32 input, INT8 output, and INT32 parameters")
        if scale_entry[4] < 16:
            raise ValueError("Mock requantize parameter buffer is too small")
        src = self._get_buffer(input_entry[0])
        dst = self._get_buffer(output_entry[0])
        params = self._get_buffer(scale_entry[0])
        multiplier = int.from_bytes(params[0:4], "little", signed=True)
        right_shift = int.from_bytes(params[4:8], "little", signed=False)
        input_zero_point = int.from_bytes(params[8:12], "little", signed=True)
        output_zero_point = int.from_bytes(params[12:16], "little", signed=True)
        logical_count = rows * columns if rows and columns else len(src) // 4
        if right_shift > 31 or logical_count > len(src) // 4:
            raise ValueError("Invalid mock requantize shape")
        tiled_output = output_entry[2] == Layout.TRANSFORMER_ACTIVATION_TILED_INT8
        if tiled_output:
            required = ((rows + 7) // 8) * columns * 8
            if not rows or not columns or required > len(dst):
                raise ValueError("Invalid tiled mock requantize shape")
            dst[:] = b"\0" * len(dst)
        elif logical_count > len(dst):
            raise ValueError("Invalid mock requantize output size")
        for idx in range(logical_count):
            value = int.from_bytes(src[idx * 4 : idx * 4 + 4], "little", signed=True)
            value = ((value - input_zero_point) * multiplier) >> right_shift
            value = max(-128, min(127, value + output_zero_point))
            dst_idx = (((idx // columns) // 8) * columns + (idx % columns)) * 8 + ((idx // columns) & 7) if tiled_output else idx
            dst[dst_idx] = value & 0xFF

    def _execute_graph_reshape(self, table, layer) -> None:
        _, input0, _, _, output, *_ = layer
        src = self._get_buffer(table[input0][0])
        dst = self._get_buffer(table[output][0])
        if len(src) != len(dst):
            raise ValueError("Mock reshape requires equal byte sizes")
        dst[:] = src

    @staticmethod
    def _signed8(value: int) -> int:
        return value if value < 128 else value - 256

    def _execute_graph_residual(self, table, layer) -> None:
        _, lhs, rhs, _, output, *_, flags, _ = layer
        a, b, dst = self._get_buffer(table[lhs][0]), self._get_buffer(table[rhs][0]), self._get_buffer(table[output][0])
        if len(a) != len(b) or len(a) != len(dst):
            raise ValueError("Residual add requires equal INT8 buffer sizes")
        ls, rs = flags & 7, (flags >> 4) & 7
        for i in range(len(dst)):
            value = (self._signed8(a[i]) >> ls) + (self._signed8(b[i]) >> rs)
            dst[i] = max(-128, min(127, value)) & 0xFF

    def _execute_graph_gelu(self, table, layer) -> None:
        _, input0, _, _, output, *_ = layer
        src, dst = self._get_buffer(table[input0][0]), self._get_buffer(table[output][0])
        if len(src) != len(dst):
            raise ValueError("GELU requires equal INT8 buffer sizes")
        for i, raw in enumerate(src):
            x = self._signed8(raw)
            if x <= -24: value = 0
            elif x >= 24: value = x
            else:
                product = x * (x + 24)
                value = int((product + 24 if product >= 0 else product - 24) / 48)
            dst[i] = value & 0xFF

    def _execute_graph_layernorm(self, table, layer) -> None:
        _, input0, _, _, output, _, bias, scale, rows, features, *_, epsilon, _ = layer
        src, dst = self._get_buffer(table[input0][0]), self._get_buffer(table[output][0])
        gamma, beta = self._get_buffer(table[scale][0]), self._get_buffer(table[bias][0])
        tiled_output = table[output][2] == Layout.TRANSFORMER_ACTIVATION_TILED_INT8
        required_output = ((rows + 7) // 8) * features * 8 if tiled_output else rows * features
        if len(src) < rows * features or len(dst) < required_output or len(gamma) < 2 * features or len(beta) < 2 * features:
            raise ValueError("Invalid LayerNorm buffers")
        if tiled_output:
            dst[:] = b"\0" * len(dst)
        for row in range(rows):
            values = [self._signed8(src[row * features + i]) for i in range(features)]
            total, squares = sum(values), sum(x * x for x in values)
            denominator = max(1, __import__("math").isqrt(features * squares - total * total + epsilon))
            for i, x in enumerate(values):
                g = int.from_bytes(gamma[2*i:2*i+2], "little", signed=True)
                b = int.from_bytes(beta[2*i:2*i+2], "little", signed=True)
                numerator = (x * features - total) << 14
                norm_q14 = (abs(numerator) // denominator) * (-1 if numerator < 0 else 1)
                scaled_q8 = ((norm_q14 * g) >> 14) + b
                rounded = scaled_q8 + 128 if scaled_q8 >= 0 else scaled_q8 - 128
                dst_idx = ((row // 8) * features + i) * 8 + (row & 7) if tiled_output else row * features + i
                dst[dst_idx] = max(-128, min(127, rounded >> 8)) & 0xFF

    def _execute_graph_activation_b_pack(self, table, layer) -> None:
        input0, output, reduction, outputs, flags = layer[1], layer[4], layer[8], layer[9], layer[22]
        src, dst = self._get_buffer(table[input0][0]), self._get_buffer(table[output][0])
        transpose = bool(flags & 1)
        source_row_major = bool(flags & 2)
        required = ((outputs + 7) // 8) * reduction * 8
        expected_input_layout = Layout.ROW_MAJOR if source_row_major else Layout.TRANSFORMER_ACTIVATION_TILED_INT8
        expected_output_layout = Layout.TRANSFORMER_ACTIVATION_TILED_INT8 if source_row_major else Layout.TRANSFORMER_DYNAMIC_B_TILED_INT8
        if table[input0][2] != expected_input_layout or table[output][2] != expected_output_layout or len(dst) < required:
            raise ValueError("Invalid dynamic B-pack buffers")
        dst[:] = b"\0" * len(dst)
        source_features = reduction if transpose else outputs
        for k in range(reduction):
            for n in range(outputs):
                source_row, source_column = (n, k) if transpose else (k, n)
                source_index = (source_row * source_features + source_column if source_row_major else
                                ((source_row // 8) * source_features + source_column) * 8 + (source_row & 7))
                output_index = ((n // 8) * reduction + k) * 8 + (n & 7)
                dst[output_index] = src[source_index]

    def _execute_graph_token_concat(self, table, layer) -> None:
        inputs, output, rows = layer[1:4], layer[4], layer[8]
        widths, total = layer[9:12], layer[12]
        dst = self._get_buffer(table[output][0])
        if len(dst) < rows * total or sum(widths) != total:
            raise ValueError("Invalid token concat output")
        for row in range(rows):
            cursor = row * total
            for index, width in zip(inputs, widths):
                if not width:
                    continue
                src = self._get_buffer(table[index][0])
                start = row * width
                dst[cursor:cursor + width] = src[start:start + width]
                cursor += width

    def _execute_graph_token_embedding(self, table, layer) -> None:
        patches_index, output_index = layer[1], layer[4]
        position_index, scale_index = layer[5], layer[7]
        patch_rows, features = layer[8], layer[9]
        patches = self._get_buffer(table[patches_index][0])
        position = self._get_buffer(table[position_index][0])
        scale = self._get_buffer(table[scale_index][0])
        output = self._get_buffer(table[output_index][0])
        patch_elements = patch_rows * features
        token_elements = (patch_rows + 1) * features
        if (len(patches) < patch_elements or len(position) < token_elements or
                len(output) < token_elements or len(scale) < 256):
            raise ValueError("Invalid token embedding buffers")
        output[:features] = position[:features]
        for index in range(patch_elements):
            scaled = self._signed8(scale[patches[index]])
            value = scaled + self._signed8(position[features + index])
            output[features + index] = max(-128, min(127, value)) & 0xFF

    def _execute_graph_softmax(self, table, layer) -> None:
        _, input0, _, _, output, _, _, _, rows, length, *_ = layer
        src, dst = self._get_buffer(table[input0][0]), self._get_buffer(table[output][0])
        lut = (32767,25520,19874,15480,12055,9388,7312,5695,4435,3454,2690,2095,1631,1270,989,770,600)
        if len(src) < rows * length or len(dst) < rows * length:
            raise ValueError("Invalid Softmax buffers")
        for row in range(rows):
            values = [self._signed8(src[row * length + i]) for i in range(length)]
            maximum = max(values)
            exps = [lut[maximum-x] if maximum-x <= 16 else 0 for x in values]
            total = sum(exps)
            for i, value in enumerate(exps):
                dst[row * length + i] = (value * 127 + total // 2) // total if total else 0

    def _execute_graph_pool(self, table, layer) -> None:
        _, input0, _, _, output, _, _, _, n, c, h, w, _, out_h, out_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, _, flags, _ = layer
        src = self._get_buffer(table[input0][0])
        dst = self._get_buffer(table[output][0])
        for batch in range(n):
            for channel in range(c):
                for oh in range(out_h):
                    for ow in range(out_w):
                        best = -128
                        total = 0
                        count = 0
                        for kh in range(kernel_h):
                            ih = oh * stride_h + kh - pad_h
                            if ih < 0 or ih >= h:
                                continue
                            for kw in range(kernel_w):
                                iw = ow * stride_w + kw - pad_w
                                if iw < 0 or iw >= w:
                                    continue
                                idx = ((batch * c + channel) * h + ih) * w + iw
                                value = src[idx] if src[idx] < 128 else src[idx] - 256
                                best = max(best, value)
                                total += value
                                count += 1
                        out_idx = ((batch * c + channel) * out_h + oh) * out_w + ow
                        result = (-((-total) // count) if total < 0 else total // count) if flags & 1 else best
                        dst[out_idx] = result & 0xFF

    def _execute_graph_upsample(self, table, layer) -> None:
        _, input0, _, _, output, _, _, _, n, c, h, w, _, out_h, out_w, *_ = layer
        input_entry = table[input0]
        output_entry = table[output]
        elem_bytes = self._dtype_bytes(input_entry[1])
        if input_entry[1] != output_entry[1]:
            raise ValueError("Mock upsample requires matching dtypes")
        src = self._get_buffer(input_entry[0])
        dst = self._get_buffer(output_entry[0])
        for batch in range(n):
            for channel in range(c):
                for oh in range(out_h):
                    ih = (oh * h) // out_h
                    for ow in range(out_w):
                        iw = (ow * w) // out_w
                        src_idx = ((batch * c + channel) * h + ih) * w + iw
                        dst_idx = ((batch * c + channel) * out_h + oh) * out_w + ow
                        dst[dst_idx * elem_bytes : dst_idx * elem_bytes + elem_bytes] = src[src_idx * elem_bytes : src_idx * elem_bytes + elem_bytes]

    def _execute_graph_concat(self, table, layer) -> None:
        _, input0, input1, input2, output, _, _, _, n, _, h, w, out_c, *_ = layer
        output_entry = table[output]
        elem_bytes = self._dtype_bytes(output_entry[1])
        dst = self._get_buffer(output_entry[0])
        channel_offset = 0
        input_indexes = [input0] + ([input1] if input1 else []) + ([input2] if input2 else [])
        for input_index in input_indexes:
            input_entry = table[input_index]
            if input_entry[1] != output_entry[1]:
                raise ValueError("Mock concat requires matching dtypes")
            src = self._get_buffer(input_entry[0])
            channels = len(src) // (n * h * w * elem_bytes)
            for batch in range(n):
                for channel in range(channels):
                    src_idx = ((batch * channels + channel) * h * w) * elem_bytes
                    dst_idx = ((batch * out_c + channel_offset + channel) * h * w) * elem_bytes
                    span = h * w * elem_bytes
                    dst[dst_idx : dst_idx + span] = src[src_idx : src_idx + span]
            channel_offset += channels
        if channel_offset != out_c:
            raise ValueError("Mock concat channel count mismatch")

    def _execute_graph_conv2d(self, table, layer) -> None:
        _, input0, _, _, output, weight, bias, _, n, c, h, w, out_c, out_h, out_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, groups, flags, _ = layer
        if groups != 1:
            raise ValueError("Mock Conv2D only supports groups=1")
        src = self._get_buffer(table[input0][0])
        wgt = self._get_buffer(table[weight][0])
        dst = self._get_buffer(table[output][0])
        bias_buf = self._get_buffer(table[bias][0]) if bias else None
        for batch in range(n):
            for oc in range(out_c):
                for oh in range(out_h):
                    for ow in range(out_w):
                        acc = 0 if bias_buf is None else int.from_bytes(bias_buf[oc * 4 : oc * 4 + 4], "little", signed=True)
                        for ic in range(c):
                            for kh in range(kernel_h):
                                ih = oh * stride_h + kh - pad_h
                                if ih < 0 or ih >= h:
                                    continue
                                for kw in range(kernel_w):
                                    iw = ow * stride_w + kw - pad_w
                                    if iw < 0 or iw >= w:
                                        continue
                                    input_idx = ((batch * c + ic) * h + ih) * w + iw
                                    k = (ic * kernel_h + kh) * kernel_w + kw
                                    if flags & CONV_FLAG_PRESLICED_WEIGHTS:
                                        k_tiles = (c * kernel_h * kernel_w + 15) // 16
                                        weight_idx = (((oc // 16) * k_tiles + (k // 16)) * 256) + (k % 16) * 16 + (oc % 16)
                                    else:
                                        weight_idx = oc * c * kernel_h * kernel_w + k
                                    a = src[input_idx] if src[input_idx] < 128 else src[input_idx] - 256
                                    b = wgt[weight_idx] if wgt[weight_idx] < 128 else wgt[weight_idx] - 256
                                    acc += a * b
                        out_idx = ((batch * out_c + oc) * out_h + oh) * out_w + ow
                        dst[out_idx * 4 : out_idx * 4 + 4] = acc.to_bytes(4, "little", signed=True)

    def _read_int8_matrix(self, buffer_id: int, rows: int, cols: int) -> list[list[int]]:
        raw = self._get_buffer(buffer_id)
        values = [byte if byte < 128 else byte - 256 for byte in raw[: rows * cols]]
        return [values[row * cols : (row + 1) * cols] for row in range(rows)]

    def _get_buffer(self, buffer_id: int) -> bytearray:
        try:
            return self.buffers[int(buffer_id)]
        except KeyError as exc:
            raise KeyError(f"Invalid mock buffer id: {buffer_id}") from exc

    @staticmethod
    def _dtype_bytes(dtype: int) -> int:
        if dtype == DType.INT8:
            return 1
        if dtype == DType.INT16:
            return 2
        if dtype == DType.INT32:
            return 4
        raise ValueError(f"Unsupported dtype: {dtype}")

    @staticmethod
    def _error_response(command: Command, request_id: int, status: Status) -> bytes:
        return b"".join(
            [
                struct.pack(
                    "<IHHIIIIII",
                    0x42435544,
                    1,
                    int(command),
                    request_id,
                    int(status),
                    RESPONSE_HEADER_STRUCT.size,
                    0,
                    int(status),
                    0,
                )
            ]
        )
