from __future__ import annotations

from functools import reduce
from operator import mul

from .dtypes import OpType, dtype_size, normalize_dtype, normalize_layout, normalize_usage
from .mock import MockBackend
from .protocol import ProtocolClient
from .serializer import pack_gemm_desc, pack_graph_exec_desc
from .tensor import Tensor


class Device:
    def __init__(self, address: str | None = None, port: int = 9000, protocol: ProtocolClient | None = None):
        self.protocol = protocol or ProtocolClient(address=address, port=port)

    @classmethod
    def mock(cls) -> "Device":
        return cls(protocol=ProtocolClient(transport=MockBackend()))

    def alloc(self, shape, dtype, layout="row_major", usage="activation") -> Tensor:
        dtype_norm = normalize_dtype(dtype)
        layout_norm = normalize_layout(layout)
        usage_norm = normalize_usage(usage)
        shape = tuple(int(dim) for dim in shape)
        size_bytes = reduce(mul, shape, 1) * dtype_size(dtype_norm)
        tensor_desc = self.protocol.tensor_desc(shape, dtype_norm, layout_norm)
        response = self.protocol.alloc(size_bytes, tensor_desc, usage_norm)
        fields = response.fields or {}
        return Tensor(
            fields["buffer_id"],
            shape,
            dtype_norm,
            layout_norm,
            fields["size_bytes"],
            self,
        )

    def write(self, tensor: Tensor, np_array, offset_bytes=0):
        raw = self._to_bytes(np_array, tensor)
        if len(raw) + int(offset_bytes) > tensor.size_bytes:
            raise ValueError("Write would exceed tensor size")
        return self.protocol.write(tensor.buffer_id, raw, offset_bytes)

    def read(self, tensor: Tensor, offset_bytes=0, nbytes=None):
        nbytes = tensor.size_bytes - int(offset_bytes) if nbytes is None else int(nbytes)
        raw = self.protocol.read(tensor.buffer_id, offset_bytes, nbytes)
        return self._from_bytes(raw, tensor, offset_bytes, nbytes)

    def execute(self, op_desc):
        op_type, descriptor = op_desc
        return self.protocol.execute(op_type, descriptor)

    def execute_gemm(self, A: Tensor, B: Tensor, C: Tensor, M: int, N: int, K: int, out_dtype="int32"):
        descriptor = pack_gemm_desc(
            A.buffer_id,
            B.buffer_id,
            C.buffer_id,
            M,
            N,
            K,
            A.dtype,
            B.dtype,
            out_dtype,
            A.layout,
            B.layout,
            C.layout,
        )
        return self.protocol.execute(OpType.GEMM, descriptor)

    def execute_graph(self, descriptor_tensor: Tensor, input_tensor: Tensor, output_tensor: Tensor, flags: int = 0):
        descriptor = pack_graph_exec_desc(
            descriptor_tensor.buffer_id,
            input_tensor.buffer_id,
            output_tensor.buffer_id,
            flags,
        )
        return self.protocol.execute(OpType.GRAPH, descriptor)

    def free(self, tensor: Tensor):
        return self.protocol.free(tensor.buffer_id)

    def load_model(self, package):
        from .compiler import upload_model

        return upload_model(self, package)

    @staticmethod
    def _to_bytes(values, tensor: Tensor) -> bytes:
        if isinstance(values, (bytes, bytearray, memoryview)):
            return bytes(values)
        if hasattr(values, "tobytes"):
            return values.tobytes(order="C")

        flat = []

        def flatten(item):
            if isinstance(item, (list, tuple)):
                for subitem in item:
                    flatten(subitem)
            else:
                flat.append(int(item))

        flatten(values)
        itemsize = dtype_size(tensor.dtype)
        return b"".join(value.to_bytes(itemsize, "little", signed=True) for value in flat)

    @staticmethod
    def _from_bytes(raw: bytes, tensor: Tensor, offset_bytes: int, nbytes: int):
        try:
            import numpy as np

            dtype = {
                "int8": np.int8,
                "int16": np.int16,
                "int32": np.int32,
            }[tensor.dtype.name.lower()]
            expected_bytes = reduce(mul, tensor.shape, 1) * np.dtype(dtype).itemsize
            if nbytes == expected_bytes and int(offset_bytes) == 0:
                return np.frombuffer(raw, dtype=dtype).reshape(tensor.shape)
            return np.frombuffer(raw, dtype=np.uint8)
        except ModuleNotFoundError:
            itemsize = dtype_size(tensor.dtype)
            return [
                int.from_bytes(raw[i : i + itemsize], "little", signed=True)
                for i in range(0, len(raw), itemsize)
            ]
