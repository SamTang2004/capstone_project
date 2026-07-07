#include "bcuda_graph.h"

#include "bcuda_status.h"

void bcuda_graph_runtime_init(BcudaGraphRuntime *runtime, BcudaAxiDma *dma) {
    if (runtime == 0) {
        return;
    }
    runtime->dma = dma;
}

int bcuda_execute_gemm(BcudaGraphRuntime *runtime, const BcudaGemmDesc *desc) {
    if (runtime == 0 || runtime->dma == 0 || desc == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }

    BcudaBuffer *A = 0;
    BcudaBuffer *B = 0;
    BcudaBuffer *C = 0;
    int status = bcuda_get_buffer(desc->A_buffer, &A);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_get_buffer(desc->B_buffer, &B);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_get_buffer(desc->C_buffer, &C);
    if (status != BC_STATUS_OK) {
        return status;
    }

    status = bcuda_axi_dma_submit_gemm(runtime->dma, desc, A, B, C);
    if (status != BC_STATUS_OK) {
        return status;
    }
    return bcuda_axi_dma_poll_done(runtime->dma);
}

int bcuda_execute_graph(BcudaGraphRuntime *runtime, const BcudaGraphExecDesc *desc) {
    if (runtime == 0 || desc == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    BcudaBuffer *model = 0;
    BcudaBuffer *input = 0;
    BcudaBuffer *output = 0;
    int status = bcuda_get_buffer(desc->model_desc_buffer, &model);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_get_buffer(desc->input_buffer, &input);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_get_buffer(desc->output_buffer, &output);
    if (status != BC_STATUS_OK) {
        return status;
    }
    (void)model;
    (void)input;
    (void)output;
    return BC_STATUS_OK;
}
