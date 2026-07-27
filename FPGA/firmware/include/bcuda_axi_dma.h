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
    BCUDA_PL_REG_RQ_OUTPUT_ZERO_POINT = 0x3cu,
    BCUDA_PL_REG_DIAG_TOTAL_CYCLES = 0x40u,
    BCUDA_PL_REG_DIAG_ENGINE_CYCLES = 0x44u,
    BCUDA_PL_REG_DIAG_COMPUTE_CYCLES = 0x48u,
    BCUDA_PL_REG_DIAG_LOAD_CYCLES = 0x4cu,
    BCUDA_PL_REG_DIAG_CACHE_WAIT_CYCLES = 0x50u,
    BCUDA_PL_REG_DIAG_CACHE_READY_CYCLES = 0x54u,
    BCUDA_PL_REG_DIAG_DDR_READ_COMMANDS = 0x58u,
    BCUDA_PL_REG_DIAG_DDR_READ_BEATS = 0x5cu,
    BCUDA_PL_REG_DIAG_DDR_WRITE_COMMANDS = 0x60u,
    BCUDA_PL_REG_DIAG_DDR_STALL_CYCLES = 0x64u,
    BCUDA_PL_REG_DIAG_CORE_ROWS = 0x68u,
    BCUDA_PL_REG_DIAG_OUTPUT_STALL_CYCLES = 0x6cu,
    BCUDA_PL_REG_DIAG_DESCRIPTOR_COUNT = 0x70u,
    BCUDA_PL_REG_DIAG_CACHE_FULL_CYCLES = 0x74u,
    BCUDA_PL_REG_DIAG_CACHE_OCCUPANCY = 0x78u,
    BCUDA_PL_REG_DIAG_MAGIC = 0x7cu
};

#define BCUDA_PL_DIAGNOSTICS_MAGIC 0x44494132u

enum {
    BCUDA_PL_CONTROL_START = 0x00000001u,
    BCUDA_PL_CONTROL_REQUANTIZE = 0x00000002u,
    BCUDA_PL_CONTROL_CONV2D = 0x00000004u,
    BCUDA_PL_CONTROL_ACTIVATION = 0x00000008u,
    BCUDA_PL_CONTROL_LEAKY_RELU = 0x00000010u,
    BCUDA_PL_CONTROL_MAXPOOL = 0x00000020u,
    BCUDA_PL_CONTROL_UPSAMPLE = 0x00000040u,
    BCUDA_PL_CONTROL_CONCAT = 0x00000080u,
    BCUDA_PL_CONTROL_AVGPOOL = 0x00000200u,
    BCUDA_PL_CONTROL_FUSED_OUTPUT = 0x00000400u,
    BCUDA_PL_CONTROL_FUSION_CONFIG = 0x00000800u,
    BCUDA_PL_CONTROL_FUSED_ACTIVATION = 0x00001000u,
    BCUDA_PL_CONTROL_INT4 = 0x00000100u,
    BCUDA_PL_STATUS_BUSY = 0x00000001u,
    BCUDA_PL_STATUS_DONE = 0x00000002u,
    BCUDA_PL_STATUS_ERROR = 0x00000004u,
    BCUDA_PL_TILE_DIM = 16u,
    BCUDA_PL_TILE_ELEMENTS = 256u,
    BCUDA_PL_ARRAY_COUNT = 1u
};

#define BCUDA_PL_INT4_TILE_DIM 8u
#define BCUDA_PL_INT4_ARRAY_COUNT 2u

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

typedef struct {
    uint32_t total_cycles;
    uint32_t engine_cycles;
    uint32_t compute_cycles;
    uint32_t load_cycles;
    uint32_t cache_wait_cycles;
    uint32_t cache_ready_cycles;
    uint32_t ddr_read_commands;
    uint32_t ddr_read_beats;
    uint32_t ddr_write_commands;
    uint32_t ddr_stall_cycles;
    uint32_t core_rows;
    uint32_t output_stall_cycles;
    uint32_t descriptor_count;
    uint32_t cache_full_cycles;
    uint8_t max_allocated_tiles;
    uint8_t max_ready_tiles;
    uint8_t allocated_tiles;
    uint8_t ready_tiles;
} BcudaPlDiagnostics;

void bcuda_axi_dma_init(BcudaAxiDma *dma, uintptr_t register_base, uint32_t timeout_cycles);
int bcuda_axi_dma_submit_gemm(BcudaAxiDma *dma, const BcudaGemmDesc *desc, BcudaBuffer *A, BcudaBuffer *B, BcudaBuffer *C);
int bcuda_axi_dma_submit_requantize(BcudaAxiDma *dma, const BcudaRequantDesc *desc, BcudaBuffer *input, BcudaBuffer *output);
int bcuda_axi_dma_submit_conv2d(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *weight, BcudaBuffer *bias, BcudaBuffer *output);
int bcuda_axi_dma_submit_fused_conv2d(BcudaAxiDma *dma, const BcudaLayerDesc *layer, const BcudaRequantParams *params, const BcudaLayerDesc *activation, BcudaBuffer *input, BcudaBuffer *weight, BcudaBuffer *bias, BcudaBuffer *output);
int bcuda_axi_dma_submit_activation(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output);
int bcuda_axi_dma_submit_maxpool(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output);
int bcuda_axi_dma_submit_avgpool(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output);
int bcuda_axi_dma_submit_upsample(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output);
int bcuda_axi_dma_submit_concat(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input0, BcudaBuffer *input1, BcudaBuffer *input2, BcudaBuffer *output);
int bcuda_axi_dma_submit_tile_load(BcudaAxiDma *dma, const BcudaTileLoadDesc *desc, BcudaBuffer *source);
int bcuda_axi_dma_submit_tiled_gemm(BcudaAxiDma *dma, const BcudaGemmDesc *desc, const BcudaTiledGemmConfig *config, BcudaBuffer *A, BcudaBuffer *B, BcudaBuffer *C);
int bcuda_axi_dma_configure_accumulator(BcudaAxiDma *dma, const BcudaAccumulatorConfig *config);
int bcuda_axi_dma_configure_memory_io(BcudaAxiDma *dma, const BcudaMemoryIoConfig *config);
int bcuda_axi_dma_poll_done(BcudaAxiDma *dma);
int bcuda_axi_dma_read_diagnostics(BcudaAxiDma *dma, BcudaPlDiagnostics *out);

#endif
