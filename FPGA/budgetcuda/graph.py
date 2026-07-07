from __future__ import annotations

from dataclasses import dataclass

from .dtypes import OpType
from .serializer import pack_gemm_desc, pack_graph_exec_desc


@dataclass(frozen=True)
class GemmOp:
    A_buffer: int
    B_buffer: int
    C_buffer: int
    M: int
    N: int
    K: int

    @property
    def op_type(self) -> OpType:
        return OpType.GEMM

    def to_descriptor(self) -> bytes:
        return pack_gemm_desc(self.A_buffer, self.B_buffer, self.C_buffer, self.M, self.N, self.K)


@dataclass(frozen=True)
class GraphExec:
    model_desc_buffer: int
    input_buffer: int
    output_buffer: int
    flags: int = 0

    @property
    def op_type(self) -> OpType:
        return OpType.GRAPH

    def to_descriptor(self) -> bytes:
        return pack_graph_exec_desc(
            self.model_desc_buffer,
            self.input_buffer,
            self.output_buffer,
            self.flags,
        )
