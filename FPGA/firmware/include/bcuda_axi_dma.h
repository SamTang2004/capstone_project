#ifndef BCUDA_AXI_DMA_H
#define BCUDA_AXI_DMA_H

#include <stdint.h>

#include "bcuda_buffer.h"

#define BCUDA_PL_DEFAULT_REGISTER_BASE 0x43C80000u

enum {
    BCUDA_PL_REG_CONTROL = 0x00u,
    BCUDA_PL_REG_STATUS = 0x04u,
    BCUDA_PL_REG_A_BASE = 0x08u,
    BCUDA_PL_REG_B_BASE = 0x0cu,
    BCUDA_PL_REG_C_BASE = 0x10u,
    BCUDA_PL_REG_A_STRIDE = 0x14u,
    BCUDA_PL_REG_B_STRIDE = 0x18u,
    BCUDA_PL_REG_C_STRIDE = 0x1cu,
    BCUDA_PL_REG_M = 0x20u,
    BCUDA_PL_REG_N = 0x24u,
    BCUDA_PL_REG_K = 0x28u,
    BCUDA_PL_REG_TILE_SIZE = 0x2cu,
    BCUDA_PL_REG_ARRAY_COUNT = 0x30u,
    BCUDA_PL_REG_RQ_MULTIPLIER = 0x34u,
    BCUDA_PL_REG_RQ_RIGHT_SHIFT = 0x38u,
    BCUDA_PL_REG_RQ_OUTPUT_ZERO_POINT = 0x3cu
};

enum {
    BCUDA_PL_CONTROL_START = 0x00000001u,
    BCUDA_PL_CONTROL_REQUANTIZE = 0x00000002u,
    BCUDA_PL_STATUS_BUSY = 0x00000001u,
    BCUDA_PL_STATUS_DONE = 0x00000002u,
    BCUDA_PL_STATUS_ERROR = 0x00000004u,
    BCUDA_PL_TILE_DIM = 16u,
    BCUDA_PL_TILE_ELEMENTS = 256u,
    BCUDA_PL_ARRAY_COUNT = 1u
};

typedef struct {
    uintptr_t register_base;
    uint32_t timeout_cycles;
    uint32_t last_control_word;
    uintptr_t last_source_address;
    uint32_t last_transfer_bytes;
    uint32_t last_tile_size;
    uint32_t last_array_count;
    uint32_t last_m_tiles;
    uint32_t last_n_tiles;
    uint32_t last_k_tiles;
    uint32_t last_n_groups;
    uint32_t last_schedule_steps;
    uint32_t last_accumulator_entries;
    uintptr_t last_memory_base;
    uint32_t last_memory_bytes;
    uint32_t last_memory_io_flags;
} BcudaAxiDma;

void bcuda_axi_dma_init(BcudaAxiDma *dma, uintptr_t register_base, uint32_t timeout_cycles);
int bcuda_axi_dma_submit_gemm(BcudaAxiDma *dma, const BcudaGemmDesc *desc, BcudaBuffer *A, BcudaBuffer *B, BcudaBuffer *C);
int bcuda_axi_dma_submit_requantize(BcudaAxiDma *dma, const BcudaRequantDesc *desc, BcudaBuffer *input, BcudaBuffer *output);
int bcuda_axi_dma_submit_tile_load(BcudaAxiDma *dma, const BcudaTileLoadDesc *desc, BcudaBuffer *source);
int bcuda_axi_dma_submit_tiled_gemm(BcudaAxiDma *dma, const BcudaGemmDesc *desc, const BcudaTiledGemmConfig *config, BcudaBuffer *A, BcudaBuffer *B, BcudaBuffer *C);
int bcuda_axi_dma_configure_accumulator(BcudaAxiDma *dma, const BcudaAccumulatorConfig *config);
int bcuda_axi_dma_configure_memory_io(BcudaAxiDma *dma, const BcudaMemoryIoConfig *config);
int bcuda_axi_dma_poll_done(BcudaAxiDma *dma);

#endif
