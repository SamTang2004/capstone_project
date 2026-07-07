#ifndef BCUDA_GRAPH_H
#define BCUDA_GRAPH_H

#include "bcuda_axi_dma.h"
#include "bcuda_protocol.h"

typedef struct {
    BcudaAxiDma *dma;
} BcudaGraphRuntime;

void bcuda_graph_runtime_init(BcudaGraphRuntime *runtime, BcudaAxiDma *dma);
int bcuda_execute_gemm(BcudaGraphRuntime *runtime, const BcudaGemmDesc *desc);
int bcuda_execute_graph(BcudaGraphRuntime *runtime, const BcudaGraphExecDesc *desc);

#endif
