#include "bcuda_axi_dma.h"

#include "bcuda_status.h"

void bcuda_axi_dma_init(BcudaAxiDma *dma, uintptr_t register_base, uint32_t timeout_cycles) {
    if (dma == 0) {
        return;
    }
    dma->register_base = register_base;
    dma->timeout_cycles = timeout_cycles;
    dma->last_control_word = 0;
}

int bcuda_axi_dma_submit_gemm(BcudaAxiDma *dma, const BcudaGemmDesc *desc, BcudaBuffer *A, BcudaBuffer *B, BcudaBuffer *C) {
    if (dma == 0 || desc == 0 || A == 0 || B == 0 || C == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (desc->A_dtype != BC_DTYPE_INT8 || desc->B_dtype != BC_DTYPE_INT8 || desc->C_dtype != BC_DTYPE_INT32) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }

    dma->last_control_word = 0x1u;
    dma->last_control_word ^= (uint32_t)(A->physical_address & 0xffffffffu);
    dma->last_control_word ^= (uint32_t)(B->physical_address & 0xffffffffu);
    dma->last_control_word ^= (uint32_t)(C->physical_address & 0xffffffffu);
    dma->last_control_word ^= desc->M ^ desc->N ^ desc->K ^ desc->flags;
    return BC_STATUS_OK;
}

int bcuda_axi_dma_poll_done(BcudaAxiDma *dma) {
    if (dma == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    return dma->timeout_cycles == 0 ? BC_STATUS_EXECUTION_FAILED : BC_STATUS_OK;
}
