#include "bcuda_axi_dma.h"

#include <stddef.h>

#include "bcuda_status.h"

static uint32_t bcuda_ceil_div_u32(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

static volatile uint32_t *bcuda_pl_regs(BcudaAxiDma *dma) {
    return dma->register_base == 0 ? 0 : (volatile uint32_t *)dma->register_base;
}

static void bcuda_pl_write(BcudaAxiDma *dma, uint32_t offset, uint32_t value) {
    volatile uint32_t *regs = bcuda_pl_regs(dma);
    if (regs != 0) {
        regs[offset / sizeof(uint32_t)] = value;
    }
}

static uint32_t bcuda_pl_read(BcudaAxiDma *dma, uint32_t offset) {
    volatile uint32_t *regs = bcuda_pl_regs(dma);
    return regs == 0 ? 0 : regs[offset / sizeof(uint32_t)];
}

int bcuda_axi_dma_read_diagnostics(BcudaAxiDma *dma, BcudaPlDiagnostics *out) {
    uint32_t occupancy;
    if (dma == 0 || out == 0 || dma->register_base == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_MAGIC) !=
        BCUDA_PL_DIAGNOSTICS_MAGIC) {
        return BC_STATUS_INVALID_COMMAND;
    }
    out->total_cycles = bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_TOTAL_CYCLES);
    out->engine_cycles = bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_ENGINE_CYCLES);
    out->compute_cycles = bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_COMPUTE_CYCLES);
    out->load_cycles = bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_LOAD_CYCLES);
    out->cache_wait_cycles =
        bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_CACHE_WAIT_CYCLES);
    out->cache_ready_cycles =
        bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_CACHE_READY_CYCLES);
    out->ddr_read_commands =
        bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_DDR_READ_COMMANDS);
    out->ddr_read_beats =
        bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_DDR_READ_BEATS);
    out->ddr_write_commands =
        bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_DDR_WRITE_COMMANDS);
    out->ddr_stall_cycles =
        bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_DDR_STALL_CYCLES);
    out->core_rows = bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_CORE_ROWS);
    out->output_stall_cycles =
        bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_OUTPUT_STALL_CYCLES);
    out->descriptor_count =
        bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_DESCRIPTOR_COUNT);
    out->cache_full_cycles =
        bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_CACHE_FULL_CYCLES);
    occupancy = bcuda_pl_read(dma, BCUDA_PL_REG_DIAG_CACHE_OCCUPANCY);
    out->max_allocated_tiles = (uint8_t)(occupancy & 0xffu);
    out->max_ready_tiles = (uint8_t)((occupancy >> 8u) & 0xffu);
    out->allocated_tiles = (uint8_t)((occupancy >> 16u) & 0xffu);
    out->ready_tiles = (uint8_t)((occupancy >> 24u) & 0xffu);
    return BC_STATUS_OK;
}

static uint32_t bcuda_stride_bytes(const BcudaBuffer *buffer, uint32_t fallback) {
    if (buffer != 0 && buffer->rank >= 2 && buffer->shape[1] != 0) {
        if (buffer->dtype == BC_DTYPE_INT4) return (buffer->shape[1] + 1u) / 2u;
        return buffer->shape[1] * (buffer->dtype == BC_DTYPE_INT32 ? 4u : 1u);
    }
    return fallback;
}

static int bcuda_validate_tiled_gemm(const BcudaGemmDesc *desc, const BcudaBuffer *A, const BcudaBuffer *B, const BcudaBuffer *C) {
    uint64_t a_required;
    uint64_t b_required;
    uint64_t c_required;
    uint32_t a_stride;
    uint32_t b_stride;
    uint32_t c_stride;

    uint32_t tile_dim = desc->A_dtype == BC_DTYPE_INT4 ? BCUDA_PL_INT4_TILE_DIM : BCUDA_PL_TILE_DIM;
    if (desc->M == 0 || desc->N == 0 || desc->K == 0 ||
        (desc->M % tile_dim) != 0 ||
        (desc->N % tile_dim) != 0 ||
        (desc->K % tile_dim) != 0) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    if (!((A->dtype == BC_DTYPE_INT8 && B->dtype == BC_DTYPE_INT8) ||
          (A->dtype == BC_DTYPE_INT4 && B->dtype == BC_DTYPE_INT4)) ||
        A->dtype != desc->A_dtype || B->dtype != desc->B_dtype || C->dtype != BC_DTYPE_INT32) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    if (desc->A_layout != BC_LAYOUT_ROW_MAJOR || desc->B_layout != BC_LAYOUT_ROW_MAJOR || desc->C_layout != BC_LAYOUT_ROW_MAJOR ||
        A->layout != BC_LAYOUT_ROW_MAJOR || B->layout != BC_LAYOUT_ROW_MAJOR || C->layout != BC_LAYOUT_ROW_MAJOR) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    uint32_t a_row_bytes = desc->A_dtype == BC_DTYPE_INT4 ? (desc->K + 1u) / 2u : desc->K;
    uint32_t b_row_bytes = desc->B_dtype == BC_DTYPE_INT4 ? (desc->N + 1u) / 2u : desc->N;
    a_stride = bcuda_stride_bytes(A, a_row_bytes);
    b_stride = bcuda_stride_bytes(B, b_row_bytes);
    c_stride = bcuda_stride_bytes(C, desc->N * sizeof(uint32_t));
    if (a_stride < a_row_bytes || b_stride < b_row_bytes || c_stride < desc->N * sizeof(uint32_t) ||
        (c_stride & 0x3u) != 0u) return BC_STATUS_SHAPE_MISMATCH;

    a_required = (uint64_t)(desc->M - 1u) * a_stride + a_row_bytes;
    b_required = (uint64_t)(desc->K - 1u) * b_stride + b_row_bytes;
    c_required = (uint64_t)(desc->M - 1u) * c_stride + (uint64_t)desc->N * sizeof(uint32_t);
    if (a_required > A->size_bytes || b_required > B->size_bytes || c_required > C->size_bytes) return BC_STATUS_SHAPE_MISMATCH;
    return BC_STATUS_OK;
}

void bcuda_axi_dma_init(BcudaAxiDma *dma, uintptr_t register_base, uint32_t timeout_cycles) {
    if (dma == 0) {
        return;
    }
    dma->register_base = register_base;
    dma->timeout_cycles = timeout_cycles;
    dma->last_control_word = 0;
    dma->last_source_address = 0;
    dma->last_transfer_bytes = 0;
    dma->last_tile_size = 0;
    dma->last_array_count = 0;
    dma->last_m_tiles = 0;
    dma->last_n_tiles = 0;
    dma->last_k_tiles = 0;
    dma->last_n_groups = 0;
    dma->last_schedule_steps = 0;
    dma->last_accumulator_entries = 0;
    dma->last_memory_base = 0;
    dma->last_memory_bytes = 0;
    dma->last_memory_io_flags = 0;
}

int bcuda_axi_dma_submit_gemm(BcudaAxiDma *dma, const BcudaGemmDesc *desc, BcudaBuffer *A, BcudaBuffer *B, BcudaBuffer *C) {
    if (dma == 0 || desc == 0 || A == 0 || B == 0 || C == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (!((desc->A_dtype == BC_DTYPE_INT8 && desc->B_dtype == BC_DTYPE_INT8) ||
          (desc->A_dtype == BC_DTYPE_INT4 && desc->B_dtype == BC_DTYPE_INT4)) ||
        desc->C_dtype != BC_DTYPE_INT32) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    int status = bcuda_validate_tiled_gemm(desc, A, B, C);
    if (status != BC_STATUS_OK) {
        return status;
    }

    uint32_t a_stride = bcuda_stride_bytes(A, desc->A_dtype == BC_DTYPE_INT4 ? (desc->K + 1u) / 2u : desc->K);
    uint32_t b_stride = bcuda_stride_bytes(B, desc->B_dtype == BC_DTYPE_INT4 ? (desc->N + 1u) / 2u : desc->N);
    uint32_t c_stride = bcuda_stride_bytes(C, desc->N * sizeof(uint32_t));

    if (dma->register_base != 0 &&
        (A->physical_address > 0xffffffffu || B->physical_address > 0xffffffffu || C->physical_address > 0xffffffffu)) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    dma->last_source_address = A->physical_address;
    dma->last_transfer_bytes = (uint32_t)(A->size_bytes + B->size_bytes + C->size_bytes);
    dma->last_tile_size = desc->A_dtype == BC_DTYPE_INT4 ? BCUDA_PL_INT4_TILE_DIM : BCUDA_PL_TILE_DIM;
    dma->last_array_count = desc->A_dtype == BC_DTYPE_INT4 ? BCUDA_PL_INT4_ARRAY_COUNT : BCUDA_PL_ARRAY_COUNT;
    dma->last_control_word = BCUDA_PL_CONTROL_START |
        (desc->A_dtype == BC_DTYPE_INT4 ? BCUDA_PL_CONTROL_INT4 : 0u);

    if (dma->register_base != 0) {
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
        bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)A->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_B_BASE, (uint32_t)B->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)C->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_A_STRIDE, a_stride);
        bcuda_pl_write(dma, BCUDA_PL_REG_B_STRIDE, b_stride);
        bcuda_pl_write(dma, BCUDA_PL_REG_C_STRIDE, c_stride);
        bcuda_pl_write(dma, BCUDA_PL_REG_M, desc->M);
        bcuda_pl_write(dma, BCUDA_PL_REG_N, desc->N);
        bcuda_pl_write(dma, BCUDA_PL_REG_K, desc->K);
        bcuda_pl_write(dma, BCUDA_PL_REG_TILE_SIZE, dma->last_tile_size);
        bcuda_pl_write(dma, BCUDA_PL_REG_ARRAY_COUNT, dma->last_array_count);
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    }
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_requantize(BcudaAxiDma *dma, const BcudaRequantDesc *desc, BcudaBuffer *input, BcudaBuffer *output) {
    if (dma == 0 || desc == 0 || input == 0 || output == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (input->dtype != BC_DTYPE_INT32 ||
        (output->dtype != BC_DTYPE_INT8 && output->dtype != BC_DTYPE_INT4)) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    if (desc->element_count == 0 || desc->right_shift > 31u || desc->input_zero_point != 0 ||
        (uint64_t)desc->element_count * sizeof(int32_t) > input->size_bytes ||
        (uint64_t)(desc->element_count + (output->dtype == BC_DTYPE_INT4 ? 1u : 0u)) /
            (output->dtype == BC_DTYPE_INT4 ? 2u : 1u) > output->size_bytes) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    if (dma->register_base != 0 && (input->physical_address > 0xffffffffu || output->physical_address > 0xffffffffu)) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    dma->last_source_address = input->physical_address;
    dma->last_transfer_bytes = desc->element_count * (uint32_t)sizeof(int32_t) +
        (output->dtype == BC_DTYPE_INT4 ? (desc->element_count + 1u) / 2u : desc->element_count);
    dma->last_control_word = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_REQUANTIZE |
        (output->dtype == BC_DTYPE_INT4 ? BCUDA_PL_CONTROL_INT4 : 0u);
    if (dma->register_base != 0) {
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
        bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)output->physical_address);
        bcuda_pl_write(dma, BCUDA_PL_REG_M, desc->element_count);
        bcuda_pl_write(dma, BCUDA_PL_REG_RQ_MULTIPLIER, (uint32_t)desc->multiplier);
        bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT, desc->right_shift);
        bcuda_pl_write(dma, BCUDA_PL_REG_RQ_OUTPUT_ZERO_POINT, (uint32_t)desc->output_zero_point);
        bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    }
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_conv2d(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *weight, BcudaBuffer *bias, BcudaBuffer *output) {
    if (dma == 0 || layer == 0 || input == 0 || weight == 0 || output == 0) return BC_STATUS_GENERAL_ERROR;
    if (!((input->dtype == BC_DTYPE_INT8 && weight->dtype == BC_DTYPE_INT8) ||
          (input->dtype == BC_DTYPE_INT4 && weight->dtype == BC_DTYPE_INT4)) ||
        output->dtype != BC_DTYPE_INT32 ||
        layer->n != 1u || layer->groups != 1u || layer->c == 0u || layer->h == 0u || layer->w == 0u || layer->out_c == 0u ||
        layer->out_h == 0u || layer->out_w == 0u || layer->kernel_h == 0u || layer->kernel_h > 16u || layer->kernel_w == 0u ||
        layer->kernel_w > 16u || layer->stride_h == 0u || layer->stride_w == 0u || layer->stride_h > 255u || layer->stride_w > 255u ||
        layer->pad_h > 255u || layer->pad_w > 255u || (bias != 0 && (bias->dtype != BC_DTYPE_INT32 || bias->size_bytes < (uint64_t)layer->out_c * sizeof(int32_t)))) return BC_STATUS_SHAPE_MISMATCH;
    if (dma->register_base == 0) return BC_STATUS_EXECUTION_FAILED;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_B_BASE, (uint32_t)weight->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)output->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_STRIDE, layer->c);
    bcuda_pl_write(dma, BCUDA_PL_REG_B_STRIDE, layer->h);
    bcuda_pl_write(dma, BCUDA_PL_REG_C_STRIDE, layer->w);
    bcuda_pl_write(dma, BCUDA_PL_REG_M, layer->out_c);
    bcuda_pl_write(dma, BCUDA_PL_REG_N, layer->out_h);
    bcuda_pl_write(dma, BCUDA_PL_REG_K, layer->out_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_TILE_SIZE, layer->kernel_h);
    bcuda_pl_write(dma, BCUDA_PL_REG_ARRAY_COUNT, layer->kernel_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_MULTIPLIER, layer->stride_h | (layer->stride_w << 8));
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT, layer->pad_h | (layer->pad_w << 8));
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_OUTPUT_ZERO_POINT, bias == 0 ? 0u : (uint32_t)bias->physical_address);
    dma->last_control_word = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_CONV2D |
        (input->dtype == BC_DTYPE_INT4 ? BCUDA_PL_CONTROL_INT4 : 0u);
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_fused_conv2d(
    BcudaAxiDma *dma,
    const BcudaLayerDesc *layer,
    const BcudaRequantParams *params,
    const BcudaLayerDesc *activation,
    BcudaBuffer *input,
    BcudaBuffer *weight,
    BcudaBuffer *bias,
    BcudaBuffer *output
) {
    uint32_t config_control;
    uint32_t leaky_shift = 0u;
    int status;
    if (dma == 0 || layer == 0 || params == 0 || input == 0 ||
        weight == 0 || output == 0 || dma->register_base == 0 ||
        input->dtype != BC_DTYPE_INT4 || weight->dtype != BC_DTYPE_INT4 ||
        output->dtype != BC_DTYPE_INT4 || params->right_shift > 31u ||
        params->input_zero_point != 0 || layer->n != 1u ||
        layer->groups != 1u || layer->out_h == 0u || layer->out_w == 0u ||
        ((uint64_t)layer->out_h * layer->out_w & 7u) != 0u ||
        (bias != 0 && (bias->dtype != BC_DTYPE_INT32 ||
         bias->size_bytes < (uint64_t)layer->out_c * sizeof(int32_t)))) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    if (input->physical_address > 0xffffffffu ||
        weight->physical_address > 0xffffffffu ||
        output->physical_address > 0xffffffffu ||
        (bias != 0 && bias->physical_address > 0xffffffffu)) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    config_control = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_INT4 |
                     BCUDA_PL_CONTROL_FUSION_CONFIG;
    if (activation != 0) {
        config_control |= BCUDA_PL_CONTROL_FUSED_ACTIVATION;
        if ((activation->flags & BCUDA_ACTIVATION_LEAKY_RELU) != 0u) {
            config_control |= BCUDA_PL_CONTROL_LEAKY_RELU;
            leaky_shift =
                (activation->flags & BCUDA_ACTIVATION_LEAKY_SHIFT_MASK) >>
                BCUDA_ACTIVATION_LEAKY_SHIFT_SHIFT;
        }
    }
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_MULTIPLIER,
                   (uint32_t)params->multiplier);
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT,
                   params->right_shift | (leaky_shift << 8));
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_OUTPUT_ZERO_POINT,
                   (uint32_t)params->output_zero_point);
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, config_control);
    status = bcuda_axi_dma_poll_done(dma);
    if (status != BC_STATUS_OK) {
        return status;
    }

    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE,
                   (uint32_t)input->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_B_BASE,
                   (uint32_t)weight->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE,
                   (uint32_t)output->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_STRIDE, layer->c);
    bcuda_pl_write(dma, BCUDA_PL_REG_B_STRIDE, layer->h);
    bcuda_pl_write(dma, BCUDA_PL_REG_C_STRIDE, layer->w);
    bcuda_pl_write(dma, BCUDA_PL_REG_M, layer->out_c);
    bcuda_pl_write(dma, BCUDA_PL_REG_N, layer->out_h);
    bcuda_pl_write(dma, BCUDA_PL_REG_K, layer->out_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_TILE_SIZE, layer->kernel_h);
    bcuda_pl_write(dma, BCUDA_PL_REG_ARRAY_COUNT, layer->kernel_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_MULTIPLIER,
                   layer->stride_h | (layer->stride_w << 8));
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT,
                   layer->pad_h | (layer->pad_w << 8));
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_OUTPUT_ZERO_POINT,
                   bias == 0 ? 0u : (uint32_t)bias->physical_address);
    dma->last_control_word = BCUDA_PL_CONTROL_START |
        BCUDA_PL_CONTROL_CONV2D | BCUDA_PL_CONTROL_INT4 |
        BCUDA_PL_CONTROL_FUSED_OUTPUT;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_activation(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output) {
    uint32_t flags;
    if (dma == 0 || layer == 0 || input == 0 || output == 0 ||
        (input->dtype != BC_DTYPE_INT8 && input->dtype != BC_DTYPE_INT4) ||
        output->dtype != input->dtype || input->size_bytes != output->size_bytes ||
        input->size_bytes == 0 || dma->register_base == 0) return BC_STATUS_SHAPE_MISMATCH;
    flags = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_ACTIVATION;
    if ((layer->flags & BCUDA_ACTIVATION_LEAKY_RELU) != 0u) flags |= BCUDA_PL_CONTROL_LEAKY_RELU;
    if (input->dtype == BC_DTYPE_INT4) flags |= BCUDA_PL_CONTROL_INT4;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)output->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_M, (uint32_t)input->size_bytes * (input->dtype == BC_DTYPE_INT4 ? 2u : 1u));
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT, (layer->flags & BCUDA_ACTIVATION_LEAKY_SHIFT_MASK) >> BCUDA_ACTIVATION_LEAKY_SHIFT_SHIFT);
    dma->last_control_word = flags;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, flags);
    return BC_STATUS_OK;
}

static uint64_t bcuda_int4_storage_bytes(uint64_t elements) {
    return (elements + 1u) / 2u;
}

static int bcuda_pl_validate_int4_tensor_op(BcudaAxiDma *dma, BcudaBuffer *input, BcudaBuffer *output) {
    if (dma == 0 || input == 0 || output == 0 || dma->register_base == 0 ||
        input->dtype != BC_DTYPE_INT4 || output->dtype != BC_DTYPE_INT4 ||
        input->physical_address > 0xffffffffu || output->physical_address > 0xffffffffu) return BC_STATUS_SHAPE_MISMATCH;
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_maxpool(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output) {
    if (layer == 0 || bcuda_pl_validate_int4_tensor_op(dma, input, output) != BC_STATUS_OK || layer->n == 0 || layer->c == 0 || layer->h == 0 || layer->w == 0 ||
        layer->out_h == 0 || layer->out_w == 0 || layer->kernel_h == 0 || layer->kernel_h > 16 || layer->kernel_w == 0 || layer->kernel_w > 16 ||
        layer->stride_h == 0 || layer->stride_h > 255 || layer->stride_w == 0 || layer->stride_w > 255 || layer->pad_h > 255 || layer->pad_w > 255 ||
        input->size_bytes < bcuda_int4_storage_bytes((uint64_t)layer->n * layer->c * layer->h * layer->w) ||
        output->size_bytes < bcuda_int4_storage_bytes((uint64_t)layer->n * layer->c * layer->out_h * layer->out_w)) return BC_STATUS_SHAPE_MISMATCH;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)output->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_STRIDE, layer->n); bcuda_pl_write(dma, BCUDA_PL_REG_B_STRIDE, layer->c); bcuda_pl_write(dma, BCUDA_PL_REG_C_STRIDE, layer->h);
    bcuda_pl_write(dma, BCUDA_PL_REG_M, layer->w); bcuda_pl_write(dma, BCUDA_PL_REG_N, layer->out_h); bcuda_pl_write(dma, BCUDA_PL_REG_K, layer->out_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_TILE_SIZE, layer->kernel_h); bcuda_pl_write(dma, BCUDA_PL_REG_ARRAY_COUNT, layer->kernel_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_MULTIPLIER, layer->stride_h | (layer->stride_w << 8)); bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT, layer->pad_h | (layer->pad_w << 8));
    dma->last_control_word = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_MAXPOOL | BCUDA_PL_CONTROL_INT4;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_avgpool(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output) {
    if (layer == 0 || bcuda_pl_validate_int4_tensor_op(dma, input, output) != BC_STATUS_OK || layer->n == 0 || layer->c == 0 || layer->h == 0 || layer->w == 0 ||
        layer->out_h == 0 || layer->out_w == 0 || layer->kernel_h == 0 || layer->kernel_h > 16 || layer->kernel_w == 0 || layer->kernel_w > 16 ||
        layer->stride_h == 0 || layer->stride_h > 255 || layer->stride_w == 0 || layer->stride_w > 255 || layer->pad_h > 255 || layer->pad_w > 255 ||
        input->size_bytes < bcuda_int4_storage_bytes((uint64_t)layer->n * layer->c * layer->h * layer->w) ||
        output->size_bytes < bcuda_int4_storage_bytes((uint64_t)layer->n * layer->c * layer->out_h * layer->out_w)) return BC_STATUS_SHAPE_MISMATCH;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)output->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_STRIDE, layer->n); bcuda_pl_write(dma, BCUDA_PL_REG_B_STRIDE, layer->c); bcuda_pl_write(dma, BCUDA_PL_REG_C_STRIDE, layer->h);
    bcuda_pl_write(dma, BCUDA_PL_REG_M, layer->w); bcuda_pl_write(dma, BCUDA_PL_REG_N, layer->out_h); bcuda_pl_write(dma, BCUDA_PL_REG_K, layer->out_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_TILE_SIZE, layer->kernel_h); bcuda_pl_write(dma, BCUDA_PL_REG_ARRAY_COUNT, layer->kernel_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_MULTIPLIER, layer->stride_h | (layer->stride_w << 8)); bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT, layer->pad_h | (layer->pad_w << 8));
    dma->last_control_word = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_AVGPOOL | BCUDA_PL_CONTROL_INT4;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_upsample(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input, BcudaBuffer *output) {
    uint32_t scale_h, scale_w;
    if (layer == 0 || bcuda_pl_validate_int4_tensor_op(dma, input, output) != BC_STATUS_OK || layer->n == 0 || layer->c == 0 || layer->h == 0 || layer->w == 0 || layer->out_h == 0 || layer->out_w == 0 ||
        layer->out_h % layer->h != 0 || layer->out_w % layer->w != 0 ||
        input->size_bytes < bcuda_int4_storage_bytes((uint64_t)layer->n * layer->c * layer->h * layer->w) ||
        output->size_bytes < bcuda_int4_storage_bytes((uint64_t)layer->n * layer->c * layer->out_h * layer->out_w)) return BC_STATUS_SHAPE_MISMATCH;
    scale_h = layer->out_h / layer->h; scale_w = layer->out_w / layer->w;
    if (scale_h == 0 || scale_h > 255 || scale_w == 0 || scale_w > 255) return BC_STATUS_SHAPE_MISMATCH;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input->physical_address); bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)output->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_STRIDE, layer->n); bcuda_pl_write(dma, BCUDA_PL_REG_B_STRIDE, layer->c); bcuda_pl_write(dma, BCUDA_PL_REG_C_STRIDE, layer->h);
    bcuda_pl_write(dma, BCUDA_PL_REG_M, layer->w); bcuda_pl_write(dma, BCUDA_PL_REG_N, layer->out_h); bcuda_pl_write(dma, BCUDA_PL_REG_K, layer->out_w);
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_MULTIPLIER, scale_h | (scale_w << 8));
    dma->last_control_word = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_UPSAMPLE | BCUDA_PL_CONTROL_INT4;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_concat(BcudaAxiDma *dma, const BcudaLayerDesc *layer, BcudaBuffer *input0, BcudaBuffer *input1, BcudaBuffer *input2, BcudaBuffer *output) {
    uint32_t c0, c1, c2;
    if (layer == 0 || input0 == 0 || input1 == 0 || output == 0 ||
        bcuda_pl_validate_int4_tensor_op(dma, input0, output) != BC_STATUS_OK ||
        input1->dtype != BC_DTYPE_INT4 || input1->physical_address > 0xffffffffu ||
        layer->n == 0 || layer->h == 0 || layer->w == 0 || layer->out_c == 0) return BC_STATUS_SHAPE_MISMATCH;
    c0 = input0->rank >= 2 ? input0->shape[1] : 0u;
    c1 = input1->rank >= 2 ? input1->shape[1] : 0u;
    c2 = input2 == 0 ? 0u : (input2->rank >= 2 ? input2->shape[1] : 0u);
    if (c0 == 0 || c1 == 0 || c0 + c1 + c2 != layer->out_c ||
        c1 > 65535 || c2 > 65535 ||
        input0->size_bytes < bcuda_int4_storage_bytes((uint64_t)layer->n * c0 * layer->h * layer->w) ||
        input1->size_bytes < bcuda_int4_storage_bytes((uint64_t)layer->n * c1 * layer->h * layer->w) ||
        (input2 != 0 && (input2->dtype != BC_DTYPE_INT4 ||
         input2->physical_address > 0xffffffffu ||
         input2->size_bytes < bcuda_int4_storage_bytes((uint64_t)layer->n * c2 * layer->h * layer->w))) ||
        output->size_bytes < bcuda_int4_storage_bytes((uint64_t)layer->n * layer->out_c * layer->h * layer->w)) return BC_STATUS_SHAPE_MISMATCH;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_BASE, (uint32_t)input0->physical_address); bcuda_pl_write(dma, BCUDA_PL_REG_B_BASE, (uint32_t)input1->physical_address); bcuda_pl_write(dma, BCUDA_PL_REG_C_BASE, (uint32_t)output->physical_address);
    bcuda_pl_write(dma, BCUDA_PL_REG_A_STRIDE, layer->n); bcuda_pl_write(dma, BCUDA_PL_REG_B_STRIDE, c0); bcuda_pl_write(dma, BCUDA_PL_REG_C_STRIDE, layer->h);
    bcuda_pl_write(dma, BCUDA_PL_REG_M, layer->w); bcuda_pl_write(dma, BCUDA_PL_REG_N, layer->h); bcuda_pl_write(dma, BCUDA_PL_REG_K, layer->w);
    bcuda_pl_write(dma, BCUDA_PL_REG_RQ_MULTIPLIER, c1); bcuda_pl_write(dma, BCUDA_PL_REG_RQ_RIGHT_SHIFT, c2); bcuda_pl_write(dma, BCUDA_PL_REG_RQ_OUTPUT_ZERO_POINT, input2 == 0 ? 0u : (uint32_t)input2->physical_address);
    dma->last_control_word = BCUDA_PL_CONTROL_START | BCUDA_PL_CONTROL_CONCAT | BCUDA_PL_CONTROL_INT4;
    bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, dma->last_control_word);
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_tile_load(BcudaAxiDma *dma, const BcudaTileLoadDesc *desc, BcudaBuffer *source) {
    if (dma == 0 || desc == 0 || source == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if ((desc->dtype != BC_DTYPE_INT8 && desc->dtype != BC_DTYPE_INT4) ||
        source->dtype != desc->dtype) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    uint32_t tile_limit = desc->dtype == BC_DTYPE_INT4 ? BCUDA_PL_INT4_TILE_DIM : BCUDA_PL_TILE_DIM;
    uint32_t row_bytes = desc->dtype == BC_DTYPE_INT4 ? (desc->tile_cols + 1u) / 2u : desc->tile_cols;
    if (desc->tile_rows == 0 || desc->tile_cols == 0 ||
        desc->tile_rows > tile_limit || desc->tile_cols > tile_limit) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    if (desc->row_stride_bytes < row_bytes) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    uint64_t last_row_offset = desc->base_offset_bytes + (uint64_t)(desc->tile_rows - 1u) * desc->row_stride_bytes;
    uint64_t final_byte = last_row_offset + row_bytes;
    if (final_byte > source->size_bytes) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    dma->last_source_address = source->physical_address + (uintptr_t)desc->base_offset_bytes;
    dma->last_transfer_bytes = desc->tile_rows * row_bytes;
    dma->last_control_word = 0x2u;
    dma->last_control_word ^= (uint32_t)(dma->last_source_address & 0xffffffffu);
    dma->last_control_word ^= desc->row_stride_bytes;
    dma->last_control_word ^= (desc->tile_rows << 16) | desc->tile_cols;
    dma->last_control_word ^= (desc->matrix_select & 1u) << 31;
    dma->last_control_word ^= desc->flags;
    return BC_STATUS_OK;
}

int bcuda_axi_dma_submit_tiled_gemm(BcudaAxiDma *dma, const BcudaGemmDesc *desc, const BcudaTiledGemmConfig *config, BcudaBuffer *A, BcudaBuffer *B, BcudaBuffer *C) {
    if (dma == 0 || desc == 0 || config == 0 || A == 0 || B == 0 || C == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (!((desc->A_dtype == BC_DTYPE_INT8 && desc->B_dtype == BC_DTYPE_INT8) ||
          (desc->A_dtype == BC_DTYPE_INT4 && desc->B_dtype == BC_DTYPE_INT4)) ||
        desc->C_dtype != BC_DTYPE_INT32 ||
        A->dtype != desc->A_dtype || B->dtype != desc->B_dtype || C->dtype != BC_DTYPE_INT32) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    uint32_t expected_tile = desc->A_dtype == BC_DTYPE_INT4 ? BCUDA_PL_INT4_TILE_DIM : BCUDA_PL_TILE_DIM;
    uint32_t expected_arrays = desc->A_dtype == BC_DTYPE_INT4 ? BCUDA_PL_INT4_ARRAY_COUNT : BCUDA_PL_ARRAY_COUNT;
    if (config->tile_size != expected_tile || config->array_count != expected_arrays) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    uint32_t m_tiles = bcuda_ceil_div_u32(desc->M, config->tile_size);
    uint32_t n_tiles = bcuda_ceil_div_u32(desc->N, config->tile_size);
    uint32_t k_tiles = bcuda_ceil_div_u32(desc->K, config->tile_size);
    uint32_t n_groups = bcuda_ceil_div_u32(n_tiles, config->array_count);
    uint32_t schedule_steps = m_tiles * n_groups * k_tiles;

    dma->last_tile_size = config->tile_size;
    dma->last_array_count = config->array_count;
    dma->last_m_tiles = m_tiles;
    dma->last_n_tiles = n_tiles;
    dma->last_k_tiles = k_tiles;
    dma->last_n_groups = n_groups;
    dma->last_schedule_steps = schedule_steps;
    dma->last_source_address = A->physical_address;
    dma->last_transfer_bytes = (uint32_t)(A->size_bytes + B->size_bytes + C->size_bytes);
    return bcuda_axi_dma_submit_gemm(dma, desc, A, B, C);
}

int bcuda_axi_dma_configure_accumulator(BcudaAxiDma *dma, const BcudaAccumulatorConfig *config) {
    if (dma == 0 || config == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (config->tile_entries == 0 || config->tile_entries > 256) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    dma->last_accumulator_entries = config->tile_entries;
    dma->last_control_word = 0x8u;
    dma->last_control_word ^= config->tile_entries;
    dma->last_control_word ^= (config->clear_on_first_pass & 1u) << 16;
    dma->last_control_word ^= config->flags;
    return BC_STATUS_OK;
}

int bcuda_axi_dma_configure_memory_io(BcudaAxiDma *dma, const BcudaMemoryIoConfig *config) {
    if (dma == 0 || config == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (config->memory_base == 0 || config->memory_bytes == 0 || (config->memory_base & 0x3u) != 0 || (config->memory_bytes & 0x3u) != 0) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    dma->last_memory_base = (uintptr_t)config->memory_base;
    dma->last_memory_bytes = config->memory_bytes;
    dma->last_memory_io_flags = config->flags;
    dma->last_control_word = 0x20u;
    dma->last_control_word ^= (uint32_t)(config->memory_base & 0xffffffffu);
    dma->last_control_word ^= config->memory_bytes;
    dma->last_control_word ^= config->flags;
    return BC_STATUS_OK;
}

int bcuda_axi_dma_poll_done(BcudaAxiDma *dma) {
    if (dma == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (dma->register_base == 0) {
        return dma->timeout_cycles == 0 ? BC_STATUS_EXECUTION_FAILED : BC_STATUS_OK;
    }

    for (uint32_t cycle = 0; cycle < dma->timeout_cycles; ++cycle) {
        uint32_t status = bcuda_pl_read(dma, BCUDA_PL_REG_STATUS);
        if ((status & BCUDA_PL_STATUS_ERROR) != 0u) {
            bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
            return BC_STATUS_EXECUTION_FAILED;
        }
        /* DONE marks the producer's final request; BUSY remains asserted
         * until the shared HP0 write path has drained it. */
        if ((status & (BCUDA_PL_STATUS_DONE | BCUDA_PL_STATUS_BUSY)) == BCUDA_PL_STATUS_DONE) {
            bcuda_pl_write(dma, BCUDA_PL_REG_CONTROL, 0u);
            return BC_STATUS_OK;
        }
    }
    return BC_STATUS_EXECUTION_FAILED;
}
