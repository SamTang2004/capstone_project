from __future__ import annotations

import struct
import time

from .diagnostics import (
    DIAG_ACCURACY,
    DIAG_MEMORY,
    DIAG_OPERATIONS,
    DIAG_TIMING,
    DIAG_UTILIZATION,
    DIAGNOSTICS_HEADER_STRUCT,
    DIAGNOSTICS_MAGIC,
    DIAGNOSTICS_RECORD_STRUCT,
    DIAGNOSTICS_VERSION,
)
from .dtypes import Command, DType, OpType, Status
from .packing import unpack_int4
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
        if not (
            desc["A_dtype"] == desc["B_dtype"]
            and desc["A_dtype"] in (DType.INT4, DType.INT8)
            and desc["C_dtype"] == DType.INT32
        ):
            raise ValueError("Mock GEMM supports matching INT4/INT8 inputs and INT32 output")
        M, N, K = desc["M"], desc["N"], desc["K"]
        A = self._read_matrix(desc["A_buffer"], M, K, desc["A_dtype"])
        B = self._read_matrix(desc["B_buffer"], K, N, desc["B_dtype"])
        out = bytearray()
        for m in range(M):
            for n in range(N):
                acc = 0
                for k in range(K):
                    acc += A[m][k] * B[k][n]
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
        diagnostics = self._diagnostics_begin(diagnostics_buffer, diagnostics_flags, num_layers)
        total_started = time.perf_counter_ns()
        for idx in range(num_layers):
            layer = LAYER_DESC_STRUCT.unpack_from(raw_model, layer_offset + idx * LAYER_DESC_STRUCT.size)
            op_type = layer[0]
            layer_started = time.perf_counter_ns()
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
            else:
                raise ValueError(f"Unsupported graph op: {op_type}")
            if diagnostics is not None:
                self._diagnostics_record(diagnostics, idx, layer, table, time.perf_counter_ns() - layer_started)
        if diagnostics is not None:
            self._diagnostics_finish(diagnostics, time.perf_counter_ns() - total_started, num_layers)

    def _diagnostics_begin(self, buffer_id: int, flags: int, num_layers: int):
        if flags == 0:
            if buffer_id != 0:
                raise ValueError("Diagnostics buffer supplied without diagnostics flags")
            return None
        raw = self._get_buffer(buffer_id)
        if len(raw) < DIAGNOSTICS_HEADER_STRUCT.size:
            raise ValueError("Diagnostics buffer is too small")
        header = list(DIAGNOSTICS_HEADER_STRUCT.unpack_from(raw))
        if (
            header[0] != DIAGNOSTICS_MAGIC
            or header[1] != DIAGNOSTICS_VERSION
            or header[2] != flags
            or header[3] < num_layers
            or header[5] != DIAGNOSTICS_RECORD_STRUCT.size
        ):
            raise ValueError("Invalid diagnostics buffer header")
        required = DIAGNOSTICS_HEADER_STRUCT.size + num_layers * DIAGNOSTICS_RECORD_STRUCT.size
        if required > len(raw):
            raise ValueError("Diagnostics records exceed their buffer")
        header[4] = 0
        header[8] = 0
        header[9] = 0
        DIAGNOSTICS_HEADER_STRUCT.pack_into(raw, 0, *header)
        return raw, header

    def _diagnostics_record(self, diagnostics, index, layer, table, measured_ns: int) -> None:
        raw, header = diagnostics
        flags, clock_hz, peak_ops = header[2], header[6], header[7]
        op_type = int(layer[0])
        input_bytes, output_bytes, weight_bytes = self._layer_buffer_bytes(layer, table)
        operations = self._layer_operations(layer, table)
        elapsed_ns = measured_ns if flags & DIAG_TIMING else 0
        cycles = elapsed_ns * clock_hz // 1_000_000_000 if flags & DIAG_TIMING else 0
        if not flags & DIAG_MEMORY:
            input_bytes = output_bytes = weight_bytes = 0
        if not flags & DIAG_OPERATIONS:
            operations = 0
        throughput = operations * 1_000_000_000 // elapsed_ns if elapsed_ns and operations else 0
        utilization_ppm = min(1_000_000, operations * 1_000_000 // (cycles * peak_ops)) if cycles and operations and flags & DIAG_UTILIZATION else 0
        accuracy = self._diagnostics_accuracy(layer, table) if flags & DIAG_ACCURACY else (0,) * 7
        offset = DIAGNOSTICS_HEADER_STRUCT.size + index * DIAGNOSTICS_RECORD_STRUCT.size
        DIAGNOSTICS_RECORD_STRUCT.pack_into(
            raw,
            offset,
            index,
            op_type,
            0,
            0,
            elapsed_ns,
            cycles,
            input_bytes,
            output_bytes,
            weight_bytes,
            operations,
            throughput,
            utilization_ppm,
            0,
            *(0,) * 14,
            *accuracy,
        )
        header[4] = index + 1
        DIAGNOSTICS_HEADER_STRUCT.pack_into(raw, 0, *header)

    def _diagnostics_accuracy(self, layer, table):
        entry = table[layer[4]]
        buffer_id, dtype, _, _, size_bytes, *_ = entry
        raw = bytes(self._get_buffer(buffer_id)[:size_bytes])
        output_hash = 1469598103934665603
        for value in raw:
            output_hash ^= value
            output_hash = (
                output_hash * 1099511628211
            ) & 0xFFFFFFFFFFFFFFFF
        if dtype == DType.INT4:
            values = [
                nibble - 16 if nibble & 8 else nibble
                for byte in raw
                for nibble in (byte & 15, byte >> 4)
            ]
            low, high = -8, 7
        elif dtype == DType.INT8:
            values = [value - 256 if value & 128 else value for value in raw]
            low, high = -128, 127
        elif dtype == DType.INT16:
            values = [
                value[0] for value in struct.iter_unpack("<h", raw)
            ]
            low, high = -32768, 32767
        elif dtype == DType.INT32:
            values = [
                value[0] for value in struct.iter_unpack("<i", raw)
            ]
            low, high = -2147483648, 2147483647
        else:
            values = []
            low = high = 0
        return (
            output_hash,
            sum(values),
            len(values),
            min(values, default=0),
            max(values, default=0),
            sum(value == 0 for value in values),
            sum(value in (low, high) for value in values),
        )

    def _diagnostics_finish(self, diagnostics, total_ns: int, layer_count: int) -> None:
        raw, header = diagnostics
        header[4] = layer_count
        header[8] = total_ns if header[2] & DIAG_TIMING else 0
        header[9] = sum(len(buffer) for buffer in self.buffers.values())
        DIAGNOSTICS_HEADER_STRUCT.pack_into(raw, 0, *header)

    @staticmethod
    def _layer_buffer_bytes(layer, table):
        op_type = layer[0]
        input_bytes = table[layer[1]][4]
        output_bytes = table[layer[4]][4]
        weight_bytes = 0
        if op_type == OpType.CONCAT:
            input_bytes += table[layer[2]][4]
            if layer[3] != 0:
                input_bytes += table[layer[3]][4]
        if op_type in (OpType.GEMM, OpType.CONV2D_GEMM):
            weight_bytes += table[layer[5]][4]
            if layer[6] != 0:
                weight_bytes += table[layer[6]][4]
        if op_type == OpType.REQUANTIZE:
            weight_bytes += table[layer[7]][4]
        return input_bytes, output_bytes, weight_bytes

    @staticmethod
    def _layer_operations(layer, table):
        op_type = layer[0]
        n, c, h, w, out_c, out_h, out_w, kernel_h, kernel_w = layer[8:17]
        output_entry = table[layer[4]]
        dtype_bits = {int(DType.INT4): 4, int(DType.INT8): 8, int(DType.INT16): 16, int(DType.INT32): 32}.get(output_entry[1], 8)
        output_elements = output_entry[4] * 8 // dtype_bits
        if op_type == OpType.GEMM:
            return 2 * n * out_c * c
        if op_type == OpType.CONV2D_GEMM:
            return 2 * n * out_h * out_w * out_c * c * kernel_h * kernel_w
        if op_type == OpType.POOL:
            return output_elements * kernel_h * kernel_w
        if op_type == OpType.REQUANTIZE:
            return output_elements * 2
        return output_elements

    def _execute_graph_gemm(self, table, layer) -> None:
        _, input0, _, _, output, weight, _, _, n, c, _, _, out_c, *_ = layer
        desc = struct.pack(
            "<IIIIIIIIIIIIII",
            table[input0][0],
            table[weight][0],
            table[output][0],
            n,
            out_c,
            c,
            table[input0][1],
            table[weight][1],
            int(DType.INT32),
            1,
            1,
            1,
            layer[22],
            0,
        )
        self._execute_gemm(desc)

    def _execute_graph_relu(self, table, layer) -> None:
        _, input0, _, _, output, *_, flags, _ = layer
        input_entry = table[input0]
        output_entry = table[output]
        src = self._get_buffer(input_entry[0])
        dst = self._get_buffer(output_entry[0])
        if input_entry[1] == DType.INT4:
            for idx in range(input_entry[4] * 2):
                value = self._read_signed(src, DType.INT4, idx)
                result = value >> ((flags >> 8) & 7) if (flags & 1) and value < 0 else max(0, value)
                self._write_signed(dst, DType.INT4, idx, result)
        elif input_entry[1] == DType.INT8:
            for idx, byte in enumerate(src[: input_entry[4]]):
                value = byte if byte < 128 else byte - 256
                dst[idx] = (value >> ((flags >> 8) & 7) if (flags & 1) and value < 0 else max(0, value)) & 0xFF
        elif input_entry[1] == DType.INT32:
            for idx in range(0, input_entry[4], 4):
                value = int.from_bytes(src[idx : idx + 4], "little", signed=True)
                value = value >> ((flags >> 8) & 7) if (flags & 1) and value < 0 else max(0, value)
                dst[idx : idx + 4] = value.to_bytes(4, "little", signed=True)
        else:
            raise ValueError("Mock ReLU supports INT4, INT8, and INT32")

    def _execute_graph_requantize(self, table, layer) -> None:
        _, input0, _, _, output, _, _, scale, *_ = layer
        input_entry = table[input0]
        output_entry = table[output]
        scale_entry = table[scale]
        if input_entry[1] != DType.INT32 or output_entry[1] not in (DType.INT4, DType.INT8) or scale_entry[1] != DType.INT32:
            raise ValueError("Mock requantize requires INT32 input and parameters with INT4/INT8 output")
        if scale_entry[4] < 16:
            raise ValueError("Mock requantize parameter buffer is too small")
        src = self._get_buffer(input_entry[0])
        dst = self._get_buffer(output_entry[0])
        params = self._get_buffer(scale_entry[0])
        multiplier = int.from_bytes(params[0:4], "little", signed=True)
        right_shift = int.from_bytes(params[4:8], "little", signed=False)
        input_zero_point = int.from_bytes(params[8:12], "little", signed=True)
        output_zero_point = int.from_bytes(params[12:16], "little", signed=True)
        output_capacity = len(dst) * (2 if output_entry[1] == DType.INT4 else 1)
        if right_shift > 31 or len(src) // 4 > output_capacity:
            raise ValueError("Invalid mock requantize shape")
        if output_entry[1] == DType.INT4:
            dst[:] = bytes(len(dst))
        for idx in range(len(src) // 4):
            value = int.from_bytes(src[idx * 4 : idx * 4 + 4], "little", signed=True)
            value = ((value - input_zero_point) * multiplier) >> right_shift
            low, high = (-8, 7) if output_entry[1] == DType.INT4 else (-128, 127)
            value = max(low, min(high, value + output_zero_point))
            self._write_signed(dst, output_entry[1], idx, value)

    def _execute_graph_reshape(self, table, layer) -> None:
        _, input0, _, _, output, *_ = layer
        src = self._get_buffer(table[input0][0])
        dst = self._get_buffer(table[output][0])
        if len(src) != len(dst):
            raise ValueError("Mock reshape requires equal byte sizes")
        dst[:] = src

    def _execute_graph_pool(self, table, layer) -> None:
        _, input0, _, _, output, _, _, _, n, c, h, w, _, out_h, out_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, _, flags, _ = layer
        input_entry = table[input0]
        output_entry = table[output]
        if input_entry[1] != output_entry[1] or input_entry[1] not in (DType.INT4, DType.INT8):
            raise ValueError("Mock pool requires matching INT4 or INT8 tensors")
        src = self._get_buffer(input_entry[0])
        dst = self._get_buffer(output_entry[0])
        if output_entry[1] == DType.INT4:
            dst[:] = bytes(len(dst))
        minimum = -8 if input_entry[1] == DType.INT4 else -128
        for batch in range(n):
            for channel in range(c):
                for oh in range(out_h):
                    for ow in range(out_w):
                        best = minimum
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
                                value = self._read_signed(src, input_entry[1], idx)
                                best = max(best, value)
                                total += value
                                count += 1
                        out_idx = ((batch * c + channel) * out_h + oh) * out_w + ow
                        result = (-((-total) // count) if total < 0 else total // count) if flags & 1 else best
                        self._write_signed(dst, output_entry[1], out_idx, result)

    def _execute_graph_upsample(self, table, layer) -> None:
        _, input0, _, _, output, _, _, _, n, c, h, w, _, out_h, out_w, *_ = layer
        input_entry = table[input0]
        output_entry = table[output]
        if input_entry[1] != output_entry[1]:
            raise ValueError("Mock upsample requires matching dtypes")
        src = self._get_buffer(input_entry[0])
        dst = self._get_buffer(output_entry[0])
        if output_entry[1] == DType.INT4:
            dst[:] = bytes(len(dst))
        for batch in range(n):
            for channel in range(c):
                for oh in range(out_h):
                    ih = (oh * h) // out_h
                    for ow in range(out_w):
                        iw = (ow * w) // out_w
                        src_idx = ((batch * c + channel) * h + ih) * w + iw
                        dst_idx = ((batch * c + channel) * out_h + oh) * out_w + ow
                        value = self._read_scalar(src, input_entry[1], src_idx)
                        self._write_scalar(dst, output_entry[1], dst_idx, value)

    def _execute_graph_concat(self, table, layer) -> None:
        _, input0, input1, input2, output, _, _, _, n, _, h, w, out_c, *_ = layer
        output_entry = table[output]
        dst = self._get_buffer(output_entry[0])
        if output_entry[1] == DType.INT4:
            dst[:] = bytes(len(dst))
        channel_offset = 0
        input_indexes = [input0] + ([input1] if input1 else []) + ([input2] if input2 else [])
        for input_index in input_indexes:
            input_entry = table[input_index]
            if input_entry[1] != output_entry[1]:
                raise ValueError("Mock concat requires matching dtypes")
            src = self._get_buffer(input_entry[0])
            logical_elements = len(src) * (2 if input_entry[1] == DType.INT4 else 1)
            channels = logical_elements // (n * h * w)
            for batch in range(n):
                for channel in range(channels):
                    for position in range(h * w):
                        src_idx = (batch * channels + channel) * h * w + position
                        dst_idx = (batch * out_c + channel_offset + channel) * h * w + position
                        value = self._read_scalar(src, input_entry[1], src_idx)
                        self._write_scalar(dst, output_entry[1], dst_idx, value)
            channel_offset += channels
        if channel_offset != out_c:
            raise ValueError("Mock concat channel count mismatch")

    def _execute_graph_conv2d(self, table, layer) -> None:
        _, input0, _, _, output, weight, bias, _, n, c, h, w, out_c, out_h, out_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, groups, *_ = layer
        if groups != 1:
            raise ValueError("Mock Conv2D only supports groups=1")
        input_entry = table[input0]
        weight_entry = table[weight]
        if input_entry[1] not in (DType.INT4, DType.INT8) or weight_entry[1] != input_entry[1]:
            raise ValueError("Mock Conv2D requires matching INT4 or INT8 input and weight tensors")
        src = self._get_buffer(input_entry[0])
        wgt = self._get_buffer(weight_entry[0])
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
                                    weight_idx = ((oc * c + ic) * kernel_h + kh) * kernel_w + kw
                                    a = self._read_signed(src, input_entry[1], input_idx)
                                    b = self._read_signed(wgt, weight_entry[1], weight_idx)
                                    acc += a * b
                        out_idx = ((batch * out_c + oc) * out_h + oh) * out_w + ow
                        dst[out_idx * 4 : out_idx * 4 + 4] = acc.to_bytes(4, "little", signed=True)

    def _read_matrix(self, buffer_id: int, rows: int, cols: int, dtype: int) -> list[list[int]]:
        raw = self._get_buffer(buffer_id)
        if dtype == DType.INT4:
            values = unpack_int4(raw, rows * cols)
        else:
            values = [byte if byte < 128 else byte - 256 for byte in raw[: rows * cols]]
        return [values[row * cols : (row + 1) * cols] for row in range(rows)]

    @staticmethod
    def _read_signed(raw, dtype: int, index: int) -> int:
        if dtype == DType.INT4:
            nibble = (raw[index // 2] >> (4 if index & 1 else 0)) & 0xF
            return nibble - 16 if nibble & 0x8 else nibble
        byte = raw[index]
        return byte - 256 if byte & 0x80 else byte

    @staticmethod
    def _write_signed(raw, dtype: int, index: int, value: int) -> None:
        if dtype == DType.INT4:
            shift = 4 if index & 1 else 0
            mask = 0xF << shift
            raw[index // 2] = (raw[index // 2] & ~mask) | ((value & 0xF) << shift)
        else:
            raw[index] = value & 0xFF

    @classmethod
    def _read_scalar(cls, raw, dtype: int, index: int) -> int:
        if dtype in (DType.INT4, DType.INT8):
            return cls._read_signed(raw, dtype, index)
        item_bytes = cls._dtype_bytes(dtype)
        offset = index * item_bytes
        return int.from_bytes(raw[offset : offset + item_bytes], "little", signed=True)

    @classmethod
    def _write_scalar(cls, raw, dtype: int, index: int, value: int) -> None:
        if dtype in (DType.INT4, DType.INT8):
            cls._write_signed(raw, dtype, index, value)
            return
        item_bytes = cls._dtype_bytes(dtype)
        offset = index * item_bytes
        raw[offset : offset + item_bytes] = int(value).to_bytes(item_bytes, "little", signed=True)

    def _get_buffer(self, buffer_id: int) -> bytearray:
        try:
            return self.buffers[int(buffer_id)]
        except KeyError as exc:
            raise KeyError(f"Invalid mock buffer id: {buffer_id}") from exc

    @staticmethod
    def _dtype_bytes(dtype: int) -> int:
        if dtype == DType.INT4:
            raise ValueError("Packed INT4 does not have an integral byte item size")
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
