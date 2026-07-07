#ifndef BCUDA_AXI_DMA_H
#define BCUDA_AXI_DMA_H

#include <stdint.h>

#include "bcuda_buffer.h"

typedef struct {
    uintptr_t register_base;
    uint32_t timeout_cycles;
    uint32_t last_control_word;
} BcudaAxiDma;

void bcuda_axi_dma_init(BcudaAxiDma *dma, uintptr_t register_base, uint32_t timeout_cycles);
int bcuda_axi_dma_submit_gemm(BcudaAxiDma *dma, const BcudaGemmDesc *desc, BcudaBuffer *A, BcudaBuffer *B, BcudaBuffer *C);
int bcuda_axi_dma_poll_done(BcudaAxiDma *dma);

#endif
