#include "bcuda_graph.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "bcuda_status.h"

static int bcuda_range_in_buffer(const BcudaBuffer *buffer, uint32_t offset, uint32_t count, uint32_t elem_size) {
    if (buffer == 0 || buffer->virtual_address == 0 || elem_size == 0) {
        return 0;
    }
    uint64_t bytes = (uint64_t)count * (uint64_t)elem_size;
    return (uint64_t)offset <= buffer->size_bytes && bytes <= buffer->size_bytes - (uint64_t)offset;
}

static int bcuda_copy_model_desc(const BcudaBuffer *buffer, BcudaModelDesc *model) {
    if (!bcuda_range_in_buffer(buffer, 0, 1, sizeof(BcudaModelDesc))) {
        return BC_STATUS_GENERAL_ERROR;
    }
    memcpy(model, buffer->virtual_address, sizeof(*model));
    if (model->magic != BCUDA_MODEL_MAGIC || model->version != BCUDA_MODEL_VERSION) {
        return BC_STATUS_INVALID_COMMAND;
    }
    if (model->num_layers == 0 || model->num_buffers == 0) {
        return BC_STATUS_INVALID_COMMAND;
    }
    if (!bcuda_range_in_buffer(buffer, model->buffer_table_offset, model->num_buffers, sizeof(BcudaModelBufferDesc))) {
        return BC_STATUS_INVALID_COMMAND;
    }
    if (!bcuda_range_in_buffer(buffer, model->layer_desc_offset, model->num_layers, sizeof(BcudaLayerDesc))) {
        return BC_STATUS_INVALID_COMMAND;
    }
    return BC_STATUS_OK;
}

static int bcuda_copy_model_buffer(const BcudaBuffer *model_buffer, const BcudaModelDesc *model, uint32_t index, BcudaModelBufferDesc *out) {
    if (model_buffer == 0 || model == 0 || out == 0 || index >= model->num_buffers) {
        return BC_STATUS_INVALID_HANDLE;
    }
    uint32_t offset = model->buffer_table_offset + index * (uint32_t)sizeof(BcudaModelBufferDesc);
    if (!bcuda_range_in_buffer(model_buffer, offset, 1, sizeof(BcudaModelBufferDesc))) {
        return BC_STATUS_INVALID_COMMAND;
    }
    memcpy(out, (const uint8_t *)model_buffer->virtual_address + offset, sizeof(*out));
    return BC_STATUS_OK;
}

static int bcuda_copy_layer(const BcudaBuffer *model_buffer, const BcudaModelDesc *model, uint32_t index, BcudaLayerDesc *out) {
    if (model_buffer == 0 || model == 0 || out == 0 || index >= model->num_layers) {
        return BC_STATUS_INVALID_COMMAND;
    }
    uint32_t offset = model->layer_desc_offset + index * (uint32_t)sizeof(BcudaLayerDesc);
    if (!bcuda_range_in_buffer(model_buffer, offset, 1, sizeof(BcudaLayerDesc))) {
        return BC_STATUS_INVALID_COMMAND;
    }
    memcpy(out, (const uint8_t *)model_buffer->virtual_address + offset, sizeof(*out));
    return BC_STATUS_OK;
}

static int bcuda_graph_table_buffer_id(const BcudaBuffer *model_buffer, const BcudaModelDesc *model, uint32_t index, uint32_t *out_id) {
    BcudaModelBufferDesc entry;
    int status = bcuda_copy_model_buffer(model_buffer, model, index, &entry);
    if (status != BC_STATUS_OK) {
        return status;
    }
    if (entry.buffer_id == 0) {
        return BC_STATUS_INVALID_HANDLE;
    }
    *out_id = entry.buffer_id;
    return BC_STATUS_OK;
}

static int bcuda_graph_table_buffer(const BcudaBuffer *model_buffer, const BcudaModelDesc *model, uint32_t index, BcudaBuffer **out) {
    uint32_t buffer_id = 0;
    int status = bcuda_graph_table_buffer_id(model_buffer, model, index, &buffer_id);
    if (status != BC_STATUS_OK) {
        return status;
    }
    return bcuda_get_buffer(buffer_id, out);
}

static uint32_t bcuda_dtype_bytes(uint32_t dtype) {
    switch (dtype) {
        case BC_DTYPE_INT8:
            return 1u;
        case BC_DTYPE_INT16:
            return 2u;
        case BC_DTYPE_INT32:
            return 4u;
        default:
            return 0u;
    }
}

static uint64_t bcuda_now_ns(void) {
    struct timespec value;
    if (timespec_get(&value, TIME_UTC) != TIME_UTC) {
        return 0;
    }
    return (uint64_t)value.tv_sec * 1000000000ull + (uint64_t)value.tv_nsec;
}

static uint64_t bcuda_diag_buffer_size(
    const BcudaBuffer *model_buffer,
    const BcudaModelDesc *model,
    uint32_t index
) {
    BcudaModelBufferDesc entry;
    if (bcuda_copy_model_buffer(model_buffer, model, index, &entry) != BC_STATUS_OK) {
        return 0;
    }
    return entry.size_bytes;
}

static uint64_t bcuda_diag_output_elements(
    const BcudaBuffer *model_buffer,
    const BcudaModelDesc *model,
    const BcudaLayerDesc *layer
) {
    BcudaModelBufferDesc output;
    if (bcuda_copy_model_buffer(model_buffer, model, layer->output, &output) != BC_STATUS_OK) {
        return 0;
    }
    uint32_t element_bytes = bcuda_dtype_bytes(output.dtype);
    return element_bytes == 0 ? 0 : output.size_bytes / element_bytes;
}

static uint64_t bcuda_diag_operations(
    const BcudaBuffer *model_buffer,
    const BcudaModelDesc *model,
    const BcudaLayerDesc *layer
) {
    uint64_t output_elements = bcuda_diag_output_elements(model_buffer, model, layer);
    switch (layer->op_type) {
        case BC_OP_GEMM:
            return 2ull * layer->n * layer->out_c * layer->c;
        case BC_OP_CONV2D_GEMM:
            return 2ull * layer->n * layer->out_h * layer->out_w * layer->out_c *
                layer->c * layer->kernel_h * layer->kernel_w;
        case BC_OP_POOL:
            return output_elements * layer->kernel_h * layer->kernel_w;
        case BC_OP_REQUANTIZE:
            return output_elements * 2ull;
        default:
            return output_elements;
    }
}

static void bcuda_diag_resources(
    const BcudaBuffer *model_buffer,
    const BcudaModelDesc *model,
    const BcudaLayerDesc *layer,
    BcudaDiagnosticsRecord *record
) {
    record->input_bytes = bcuda_diag_buffer_size(model_buffer, model, layer->input0);
    record->output_bytes = bcuda_diag_buffer_size(model_buffer, model, layer->output);
    if (layer->op_type == BC_OP_CONCAT) {
        record->input_bytes += bcuda_diag_buffer_size(model_buffer, model, layer->input1);
        if (layer->input2 != 0) {
            record->input_bytes += bcuda_diag_buffer_size(model_buffer, model, layer->input2);
        }
    }
    if (layer->op_type == BC_OP_GEMM || layer->op_type == BC_OP_CONV2D_GEMM) {
        record->weight_bytes = bcuda_diag_buffer_size(model_buffer, model, layer->weight);
        if (layer->bias != 0) {
            record->weight_bytes += bcuda_diag_buffer_size(model_buffer, model, layer->bias);
        }
        if (layer->scale != 0) {
            record->weight_bytes += bcuda_diag_buffer_size(model_buffer, model, layer->scale);
        }
    } else if (layer->op_type == BC_OP_REQUANTIZE) {
        record->weight_bytes = bcuda_diag_buffer_size(model_buffer, model, layer->scale);
    }
}

static int bcuda_diag_open(
    const BcudaGraphExecDesc *desc,
    uint32_t layer_count,
    BcudaBuffer **buffer_out,
    BcudaDiagnosticsHeader **header_out,
    BcudaDiagnosticsRecord **records_out
) {
    *buffer_out = 0;
    *header_out = 0;
    *records_out = 0;
    if (desc->flags == 0) {
        return desc->diagnostics_buffer == 0 ? BC_STATUS_OK : BC_STATUS_SHAPE_MISMATCH;
    }
    if ((desc->flags & ~BCUDA_DIAG_ALL) != 0 || desc->diagnostics_buffer == 0) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    BcudaBuffer *buffer = 0;
    int status = bcuda_get_buffer(desc->diagnostics_buffer, &buffer);
    if (status != BC_STATUS_OK) {
        return status;
    }
    if (buffer->virtual_address == 0 || buffer->size_bytes < sizeof(BcudaDiagnosticsHeader)) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    BcudaDiagnosticsHeader *header = (BcudaDiagnosticsHeader *)buffer->virtual_address;
    uint64_t required = sizeof(*header) + (uint64_t)layer_count * sizeof(BcudaDiagnosticsRecord);
    if (header->magic != BCUDA_DIAGNOSTICS_MAGIC || header->version != BCUDA_DIAGNOSTICS_VERSION ||
        header->flags != desc->flags || header->layer_capacity < layer_count ||
        header->record_bytes != sizeof(BcudaDiagnosticsRecord) || header->clock_hz == 0 ||
        header->peak_operations_per_cycle == 0 || required > buffer->size_bytes) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    if ((desc->flags & BCUDA_DIAG_UTILIZATION) != 0 &&
        (desc->flags & (BCUDA_DIAG_TIMING | BCUDA_DIAG_OPERATIONS)) !=
            (BCUDA_DIAG_TIMING | BCUDA_DIAG_OPERATIONS)) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    header->layer_count = 0;
    header->total_elapsed_ns = 0;
    header->arena_bytes_used = 0;
    BcudaDiagnosticsRecord *records = (BcudaDiagnosticsRecord *)((uint8_t *)buffer->virtual_address + sizeof(*header));
    memset(records, 0, (size_t)layer_count * sizeof(*records));
    *buffer_out = buffer;
    *header_out = header;
    *records_out = records;
    return BC_STATUS_OK;
}

static uint64_t bcuda_diag_rate(uint64_t operations, uint64_t elapsed_ns) {
    if (operations == 0 || elapsed_ns == 0) {
        return 0;
    }
    long double value = (long double)operations * 1000000000.0L / (long double)elapsed_ns;
    return value >= (long double)ULLONG_MAX ? ULLONG_MAX : (uint64_t)value;
}

static uint32_t bcuda_diag_utilization(uint64_t operations, uint64_t cycles, uint32_t peak_operations_per_cycle) {
    if (operations == 0 || cycles == 0 || peak_operations_per_cycle == 0) {
        return 0;
    }
    long double value = (long double)operations * 1000000.0L /
        ((long double)cycles * peak_operations_per_cycle);
    if (value >= 1000000.0L) {
        return 1000000u;
    }
    return (uint32_t)value;
}

static uint32_t bcuda_round_up_tile(uint32_t value) {
    return (value + 15u) & ~15u;
}

static uint32_t bcuda_row_stride(const BcudaBuffer *buffer, uint32_t fallback, uint32_t element_bytes) {
    if (buffer != 0 && buffer->rank >= 2 && buffer->shape[1] != 0) {
        return buffer->shape[1] * element_bytes;
    }
    return fallback;
}

static int bcuda_gemm_buffer_fits(const BcudaBuffer *buffer, uint32_t rows, uint32_t row_bytes, uint32_t stride) {
    uint64_t required;
    if (buffer == 0 || rows == 0 || stride < row_bytes) return 0;
    required = (uint64_t)(rows - 1u) * stride + row_bytes;
    return required <= buffer->size_bytes;
}

static BcudaTensorDesc bcuda_tensor_desc_2d(uint32_t dtype, uint32_t rows, uint32_t cols) {
    BcudaTensorDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.dtype = dtype;
    desc.rank = 2;
    desc.dim0 = rows;
    desc.dim1 = cols;
    desc.layout = BC_LAYOUT_ROW_MAJOR;
    return desc;
}

static int bcuda_execute_padded_gemm(BcudaGraphRuntime *runtime, const BcudaGemmDesc *desc,
                                     BcudaBuffer *A, BcudaBuffer *B, BcudaBuffer *C) {
    BcudaBuffer *padded_a = 0;
    BcudaBuffer *padded_b = 0;
    BcudaBuffer *padded_c = 0;
    BcudaGemmDesc padded_desc;
    uint32_t padded_m = bcuda_round_up_tile(desc->M);
    uint32_t padded_n = bcuda_round_up_tile(desc->N);
    uint32_t padded_k = bcuda_round_up_tile(desc->K);
    uint32_t a_stride = bcuda_row_stride(A, desc->K, 1);
    uint32_t b_stride = bcuda_row_stride(B, desc->N, 1);
    uint32_t c_stride = bcuda_row_stride(C, desc->N * sizeof(int32_t), sizeof(int32_t));
    uint32_t a_id = 0, b_id = 0, c_id = 0;
    int status;

    if (padded_m == 0 || padded_n == 0 || padded_k == 0 ||
        !bcuda_gemm_buffer_fits(A, desc->M, desc->K, a_stride) ||
        !bcuda_gemm_buffer_fits(B, desc->K, desc->N, b_stride) ||
        !bcuda_gemm_buffer_fits(C, desc->M, desc->N * sizeof(int32_t), c_stride)) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    status = bcuda_alloc_buffer((uint64_t)padded_m * padded_k,
                                bcuda_tensor_desc_2d(BC_DTYPE_INT8, padded_m, padded_k), A->usage, &a_id);
    if (status != BC_STATUS_OK) goto done;
    status = bcuda_alloc_buffer((uint64_t)padded_k * padded_n,
                                bcuda_tensor_desc_2d(BC_DTYPE_INT8, padded_k, padded_n), B->usage, &b_id);
    if (status != BC_STATUS_OK) goto done;
    status = bcuda_alloc_buffer((uint64_t)padded_m * padded_n * sizeof(int32_t),
                                bcuda_tensor_desc_2d(BC_DTYPE_INT32, padded_m, padded_n), C->usage, &c_id);
    if (status != BC_STATUS_OK) goto done;
    if (bcuda_get_buffer(a_id, &padded_a) != BC_STATUS_OK ||
        bcuda_get_buffer(b_id, &padded_b) != BC_STATUS_OK ||
        bcuda_get_buffer(c_id, &padded_c) != BC_STATUS_OK) {
        status = BC_STATUS_GENERAL_ERROR;
        goto done;
    }

    memset(padded_a->virtual_address, 0, (size_t)padded_a->size_bytes);
    memset(padded_b->virtual_address, 0, (size_t)padded_b->size_bytes);
    memset(padded_c->virtual_address, 0, (size_t)padded_c->size_bytes);
    for (uint32_t row = 0; row < desc->M; row++) {
        memcpy((uint8_t *)padded_a->virtual_address + (uint64_t)row * padded_k,
               (const uint8_t *)A->virtual_address + (uint64_t)row * a_stride, desc->K);
    }
    for (uint32_t row = 0; row < desc->K; row++) {
        memcpy((uint8_t *)padded_b->virtual_address + (uint64_t)row * padded_n,
               (const uint8_t *)B->virtual_address + (uint64_t)row * b_stride, desc->N);
    }

    padded_desc = *desc;
    padded_desc.M = padded_m;
    padded_desc.N = padded_n;
    padded_desc.K = padded_k;
    status = bcuda_axi_dma_submit_gemm(runtime->dma, &padded_desc, padded_a, padded_b, padded_c);
    if (status == BC_STATUS_OK) status = bcuda_axi_dma_poll_done(runtime->dma);
    if (status == BC_STATUS_OK) {
        for (uint32_t row = 0; row < desc->M; row++) {
            memcpy((uint8_t *)C->virtual_address + (uint64_t)row * c_stride,
                   (const uint8_t *)padded_c->virtual_address + (uint64_t)row * padded_n * sizeof(int32_t),
                   (size_t)desc->N * sizeof(int32_t));
        }
    }

done:
    if (c_id != 0) bcuda_free_buffer(c_id);
    if (b_id != 0) bcuda_free_buffer(b_id);
    if (a_id != 0) bcuda_free_buffer(a_id);
    return status;
}

static int32_t bcuda_read_i32(const uint8_t *base) {
    int32_t value = 0;
    memcpy(&value, base, sizeof(value));
    return value;
}

static void bcuda_write_i32(uint8_t *base, int32_t value) {
    memcpy(base, &value, sizeof(value));
}

static int8_t bcuda_requantize_value(int32_t value, const BcudaRequantParams *params) {
    int64_t scaled = (((int64_t)value - params->input_zero_point) * params->multiplier) >> params->right_shift;
    scaled += params->output_zero_point;
    if (scaled < -128) {
        return -128;
    }
    if (scaled > 127) {
        return 127;
    }
    return (int8_t)scaled;
}

static int bcuda_execute_gemm_layer(BcudaGraphRuntime *runtime, const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer, uint64_t *bias_elapsed_ns) {
    BcudaBuffer *input = 0;
    BcudaBuffer *weight = 0;
    BcudaBuffer *output = 0;
    BcudaBuffer *bias = 0;
    BcudaBuffer *scale = 0;
    BcudaRequantParams params;
    uint32_t fused_requant = layer->scale != 0u;
    int status = bcuda_graph_table_buffer(model_buffer, model, layer->input0, &input);
    if (status != BC_STATUS_OK) {
        return status;
    }
    if (layer->bias != 0u) {
        status = bcuda_graph_table_buffer(model_buffer, model, layer->bias, &bias);
        if (status != BC_STATUS_OK) return status;
    }
    status = bcuda_graph_table_buffer(model_buffer, model, layer->weight, &weight);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_graph_table_buffer(model_buffer, model, layer->output, &output);
    if (status != BC_STATUS_OK) {
        return status;
    }
    if (fused_requant) {
        status = bcuda_graph_table_buffer(model_buffer, model, layer->scale, &scale);
        if (status != BC_STATUS_OK) return status;
        if (scale->dtype != BC_DTYPE_INT32 || scale->size_bytes < sizeof(params) ||
            output->dtype != BC_DTYPE_INT8 ||
            (output->layout != BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8 &&
             output->layout != BC_LAYOUT_TRANSFORMER_DYNAMIC_B_TILED_INT8 &&
             output->layout != BC_LAYOUT_ROW_MAJOR)) {
            return BC_STATUS_SHAPE_MISMATCH;
        }
        memcpy(&params, scale->virtual_address, sizeof(params));
    }

    BcudaGemmDesc gemm;
    memset(&gemm, 0, sizeof(gemm));
    gemm.A_buffer = input->buffer_id;
    gemm.B_buffer = weight->buffer_id;
    gemm.C_buffer = output->buffer_id;
    gemm.M = layer->n;
    gemm.N = layer->out_c;
    gemm.K = layer->c;
    gemm.A_dtype = BC_DTYPE_INT8;
    gemm.B_dtype = BC_DTYPE_INT8;
    gemm.C_dtype = fused_requant ? BC_DTYPE_INT8 : BC_DTYPE_INT32;
    gemm.A_layout = input->layout;
    gemm.B_layout = weight->layout;
    gemm.C_layout = output->layout;
    gemm.flags = layer->flags;
    if (fused_requant) {
        if (input->layout != BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8 ||
            (weight->layout != BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT8 &&
             weight->layout != BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT4 &&
             weight->layout != BC_LAYOUT_TRANSFORMER_DYNAMIC_B_TILED_INT8)) {
            return BC_STATUS_SHAPE_MISMATCH;
        }
        status = bcuda_axi_dma_submit_transformer_gemm_requant(
            runtime->dma, &gemm, &params, input, weight, bias, output,
            weight->layout == BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT4);
        return status == BC_STATUS_OK ? bcuda_axi_dma_poll_done(runtime->dma) : status;
    }

    status = bcuda_execute_gemm(runtime, &gemm);
    if (status == BC_STATUS_OK && bias != 0) {
        uint64_t bias_started_ns = bcuda_now_ns();
        if (bias->dtype != BC_DTYPE_INT32 || bias->size_bytes < (uint64_t)layer->out_c * sizeof(int32_t) ||
            output->dtype != BC_DTYPE_INT32 || output->size_bytes < (uint64_t)layer->n * layer->out_c * sizeof(int32_t)) {
            return BC_STATUS_SHAPE_MISMATCH;
        }
        /* XRT allocations and every INT32 graph offset are word aligned. Keep
         * this as a dense row loop so ARM NEON can vectorize it; board builds
         * also divide independent rows across the two Cortex-A9 cores. */
        int32_t *destination = (int32_t *)output->virtual_address;
        const int32_t *bias_values = (const int32_t *)bias->virtual_address;
        uint32_t rows = layer->n;
        uint32_t columns = layer->out_c;
        #pragma omp parallel for if(rows >= 8u)
        for (uint32_t row = 0; row < rows; row++) {
            int32_t *row_values = destination + (uint64_t)row * columns;
            for (uint32_t column = 0; column < columns; column++) {
                row_values[column] += bias_values[column];
            }
        }
        if (bias_elapsed_ns != 0) {
            uint64_t bias_finished_ns = bcuda_now_ns();
            *bias_elapsed_ns = bias_finished_ns >= bias_started_ns
                ? bias_finished_ns - bias_started_ns : 0;
        }
    }
    return status;
}

static int bcuda_execute_activation_layer(BcudaGraphRuntime *runtime, const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer) {
    BcudaBuffer *input = 0;
    BcudaBuffer *output = 0;
    int status = bcuda_graph_table_buffer(model_buffer, model, layer->input0, &input);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_graph_table_buffer(model_buffer, model, layer->output, &output);
    if (status != BC_STATUS_OK) {
        return status;
    }
    if (input->dtype != output->dtype || input->size_bytes > output->size_bytes) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    if (input->dtype != BC_DTYPE_INT8 && input->dtype != BC_DTYPE_INT32) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    if (input->dtype == BC_DTYPE_INT8 && runtime != 0 && runtime->dma != 0 && runtime->dma->register_base != 0) {
        status = bcuda_axi_dma_submit_activation(runtime->dma, layer, input, output);
        return status == BC_STATUS_OK ? bcuda_axi_dma_poll_done(runtime->dma) : status;
    }
    uint32_t leaky = layer->flags & BCUDA_ACTIVATION_LEAKY_RELU;
    uint32_t shift = (layer->flags & BCUDA_ACTIVATION_LEAKY_SHIFT_MASK) >> BCUDA_ACTIVATION_LEAKY_SHIFT_SHIFT;

    if (input->dtype == BC_DTYPE_INT8) {
        const int8_t *src = (const int8_t *)input->virtual_address;
        int8_t *dst = (int8_t *)output->virtual_address;
        for (uint64_t i = 0; i < input->size_bytes; i++) {
            dst[i] = src[i] < 0 ? (leaky ? (src[i] >> shift) : 0) : src[i];
        }
    } else {
        uint32_t elements = (uint32_t)(input->size_bytes / sizeof(int32_t));
        const uint8_t *src = (const uint8_t *)input->virtual_address;
        uint8_t *dst = (uint8_t *)output->virtual_address;
        for (uint32_t i = 0; i < elements; i++) {
            int32_t value = bcuda_read_i32(src + i * sizeof(int32_t));
            bcuda_write_i32(dst + i * sizeof(int32_t), value < 0 ? (leaky ? (value >> shift) : 0) : value);
        }
    }
    return BC_STATUS_OK;
}

static int bcuda_execute_requantize_layer(BcudaGraphRuntime *runtime, const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer) {
    BcudaBuffer *input = 0;
    BcudaBuffer *output = 0;
    BcudaBuffer *scale = 0;
    int status = bcuda_graph_table_buffer(model_buffer, model, layer->input0, &input);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_graph_table_buffer(model_buffer, model, layer->output, &output);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_graph_table_buffer(model_buffer, model, layer->scale, &scale);
    if (status != BC_STATUS_OK) {
        return status;
    }
    if (input->dtype != BC_DTYPE_INT32 || output->dtype != BC_DTYPE_INT8 || scale->dtype != BC_DTYPE_INT32 ||
        scale->size_bytes < sizeof(BcudaRequantParams)) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    BcudaRequantParams params;
    memcpy(&params, scale->virtual_address, sizeof(params));
    uint32_t elements = (uint32_t)(input->size_bytes / sizeof(int32_t));
    if (elements == 0 || params.right_shift > 31u) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    uint32_t packed_output = output->layout == BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8;
    if (packed_output && (input->layout != BC_LAYOUT_ROW_MAJOR || layer->n == 0 || layer->c == 0 ||
        (uint64_t)layer->n * layer->c != elements)) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    uint64_t required_output_bytes = packed_output
        ? (((uint64_t)layer->n + 7u) / 8u) * layer->c * 8u
        : elements;
    if (required_output_bytes > output->size_bytes ||
        (!packed_output && output->layout != BC_LAYOUT_ROW_MAJOR)) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    if (runtime != 0 && runtime->dma != 0 && runtime->dma->register_base != 0) {
        BcudaRequantDesc desc;
        memset(&desc, 0, sizeof(desc));
        desc.input_buffer = input->buffer_id;
        desc.output_buffer = output->buffer_id;
        desc.element_count = elements;
        desc.multiplier = params.multiplier;
        desc.right_shift = params.right_shift;
        desc.input_zero_point = params.input_zero_point;
        desc.output_zero_point = params.output_zero_point;
        status = packed_output
            ? bcuda_axi_dma_submit_transformer_requantize_pack(runtime->dma, &desc, input, output, layer->n, layer->c)
            : bcuda_axi_dma_submit_requantize(runtime->dma, &desc, input, output);
        return status == BC_STATUS_OK ? bcuda_axi_dma_poll_done(runtime->dma) : status;
    }

    const uint8_t *src = (const uint8_t *)input->virtual_address;
    int8_t *dst = (int8_t *)output->virtual_address;
    if (packed_output) {
        memset(dst, 0, (size_t)output->size_bytes);
        for (uint32_t row = 0; row < layer->n; ++row) {
            for (uint32_t column = 0; column < layer->c; ++column) {
                uint32_t input_index = row * layer->c + column;
                uint64_t output_index = ((uint64_t)(row / 8u) * layer->c + column) * 8u + (row & 7u);
                dst[output_index] = bcuda_requantize_value(
                    bcuda_read_i32(src + input_index * sizeof(int32_t)), &params);
            }
        }
    } else {
        for (uint32_t i = 0; i < elements; i++) {
            dst[i] = bcuda_requantize_value(bcuda_read_i32(src + i * sizeof(int32_t)), &params);
        }
    }
    return BC_STATUS_OK;
}

static int bcuda_execute_reshape_layer(const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer) {
    BcudaBuffer *input = 0;
    BcudaBuffer *output = 0;
    int status = bcuda_graph_table_buffer(model_buffer, model, layer->input0, &input);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_graph_table_buffer(model_buffer, model, layer->output, &output);
    if (status != BC_STATUS_OK) {
        return status;
    }
    if (input->size_bytes != output->size_bytes) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    if (input->buffer_id != output->buffer_id) {
        memmove(output->virtual_address, input->virtual_address, (size_t)input->size_bytes);
    }
    return BC_STATUS_OK;
}

static int bcuda_execute_activation_b_pack_layer(BcudaGraphRuntime *runtime, const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer) {
    BcudaBuffer *input = 0, *output = 0;
    int status = bcuda_graph_table_buffer(model_buffer, model, layer->input0, &input);
    if (status == BC_STATUS_OK) status = bcuda_graph_table_buffer(model_buffer, model, layer->output, &output);
    if (status != BC_STATUS_OK) return status;
    uint32_t reduction = layer->n, outputs = layer->c, transpose = layer->flags & 1u;
    uint32_t source_row_major = (layer->flags >> 1) & 1u;
    uint32_t source_rows = transpose ? outputs : reduction;
    uint32_t source_features = transpose ? reduction : outputs;
    uint64_t input_required = source_row_major ? (uint64_t)source_rows * source_features :
        ((uint64_t)(source_rows + 7u) / 8u) * source_features * 8u;
    uint64_t output_required = ((uint64_t)(outputs + 7u) / 8u) * reduction * 8u;
    uint32_t expected_output_layout = source_row_major ? BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8 : BC_LAYOUT_TRANSFORMER_DYNAMIC_B_TILED_INT8;
    if (reduction == 0 || outputs == 0 || (layer->flags & ~3u) != 0 ||
        input->dtype != BC_DTYPE_INT8 || output->dtype != BC_DTYPE_INT8 ||
        input->layout != (source_row_major ? BC_LAYOUT_ROW_MAJOR : BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8) ||
        output->layout != expected_output_layout ||
        input_required > input->size_bytes || output_required > output->size_bytes) return BC_STATUS_SHAPE_MISMATCH;
    if (runtime != 0 && runtime->dma != 0 && runtime->dma->register_base != 0) {
        status = bcuda_axi_dma_submit_transformer_b_pack(runtime->dma, input, output, reduction, outputs, transpose, source_row_major);
        return status == BC_STATUS_OK ? bcuda_axi_dma_poll_done(runtime->dma) : status;
    }
    const uint8_t *src = (const uint8_t *)input->virtual_address;
    uint8_t *dst = (uint8_t *)output->virtual_address;
    memset(dst, 0, (size_t)output_required);
    for (uint32_t k = 0; k < reduction; ++k) {
        for (uint32_t n = 0; n < outputs; ++n) {
            uint32_t source_row = transpose ? n : k;
            uint32_t source_column = transpose ? k : n;
            uint64_t source_index = source_row_major
                ? (uint64_t)source_row * source_features + source_column
                : ((uint64_t)(source_row / 8u) * source_features + source_column) * 8u + (source_row & 7u);
            uint64_t output_index = ((uint64_t)(n / 8u) * reduction + k) * 8u + (n & 7u);
            dst[output_index] = src[source_index];
        }
    }
    return BC_STATUS_OK;
}

static int bcuda_execute_token_concat_layer(const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer) {
    BcudaBuffer *inputs[3] = {0, 0, 0}, *output = 0;
    uint32_t indices[3] = {layer->input0, layer->input1, layer->input2};
    uint32_t widths[3] = {layer->c, layer->h, layer->w};
    uint64_t total = (uint64_t)widths[0] + widths[1] + widths[2];
    if (layer->n == 0 || total == 0 || total != layer->out_c) return BC_STATUS_SHAPE_MISMATCH;
    int status = bcuda_graph_table_buffer(model_buffer, model, layer->output, &output);
    if (status != BC_STATUS_OK) return status;
    int tiled_output = output->layout == BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8;
    uint64_t required_output = tiled_output
        ? ((uint64_t)(layer->n + 7u) / 8u) * total * 8u
        : (uint64_t)layer->n * total;
    if (output->dtype != BC_DTYPE_INT8 ||
        (output->layout != BC_LAYOUT_ROW_MAJOR && !tiled_output) ||
        required_output > output->size_bytes) return BC_STATUS_SHAPE_MISMATCH;
    for (uint32_t part = 0; part < 3; ++part) {
        if (widths[part] == 0) continue;
        status = bcuda_graph_table_buffer(model_buffer, model, indices[part], &inputs[part]);
        if (status != BC_STATUS_OK) return status;
        if (inputs[part]->dtype != BC_DTYPE_INT8 || inputs[part]->layout != BC_LAYOUT_ROW_MAJOR ||
            (uint64_t)layer->n * widths[part] > inputs[part]->size_bytes) return BC_STATUS_SHAPE_MISMATCH;
    }
    uint8_t *dst = (uint8_t *)output->virtual_address;
    for (uint32_t row = 0; row < layer->n; ++row) {
        uint32_t cursor = 0;
        for (uint32_t part = 0; part < 3; ++part) {
            if (widths[part] == 0) continue;
            if (!tiled_output) {
                memcpy(dst + (uint64_t)row * total + cursor,
                       (const uint8_t *)inputs[part]->virtual_address + (uint64_t)row * widths[part], widths[part]);
            } else {
                const uint8_t *src = (const uint8_t *)inputs[part]->virtual_address +
                    (uint64_t)row * widths[part];
                for (uint32_t column = 0; column < widths[part]; ++column) {
                    uint64_t output_column = cursor + column;
                    uint64_t destination = ((uint64_t)(row / 8u) * total + output_column) * 8u +
                        (row & 7u);
                    dst[destination] = src[column];
                }
            }
            cursor += widths[part];
        }
    }
    return BC_STATUS_OK;
}

static int8_t bcuda_saturate_int8(int32_t value) {
    if (value > 127) return 127;
    if (value < -128) return -128;
    return (int8_t)value;
}

static int bcuda_execute_token_embedding_layer(const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer) {
    BcudaBuffer *patches = 0, *position = 0, *scale = 0, *output = 0;
    int status = bcuda_graph_table_buffer(model_buffer, model, layer->input0, &patches);
    if (status == BC_STATUS_OK) status = bcuda_graph_table_buffer(model_buffer, model, layer->weight, &position);
    if (status == BC_STATUS_OK) status = bcuda_graph_table_buffer(model_buffer, model, layer->scale, &scale);
    if (status == BC_STATUS_OK) status = bcuda_graph_table_buffer(model_buffer, model, layer->output, &output);
    if (status != BC_STATUS_OK) return status;
    uint64_t patch_elements = (uint64_t)layer->n * layer->c;
    uint64_t token_elements = (uint64_t)(layer->n + 1u) * layer->c;
    if (layer->n == 0 || layer->c == 0 || patches->dtype != BC_DTYPE_INT8 ||
        position->dtype != BC_DTYPE_INT8 || output->dtype != BC_DTYPE_INT8 ||
        scale->dtype != BC_DTYPE_INT8 || patches->layout != BC_LAYOUT_ROW_MAJOR ||
        position->layout != BC_LAYOUT_ROW_MAJOR || output->layout != BC_LAYOUT_ROW_MAJOR ||
        patches->size_bytes < patch_elements || position->size_bytes < token_elements ||
        output->size_bytes < token_elements || scale->size_bytes < 256u) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    const int8_t *src = (const int8_t *)patches->virtual_address;
    const int8_t *pos = (const int8_t *)position->virtual_address;
    const int8_t *lut = (const int8_t *)scale->virtual_address;
    int8_t *dst = (int8_t *)output->virtual_address;
    memcpy(dst, pos, layer->c);
    for (uint64_t i = 0; i < patch_elements; ++i) {
        int8_t scaled = lut[(uint8_t)src[i]];
        dst[layer->c + i] = bcuda_saturate_int8((int32_t)scaled + pos[layer->c + i]);
    }
    return BC_STATUS_OK;
}

static uint32_t bcuda_integer_sqrt(uint64_t value) {
    uint64_t op = value, result = 0, one = 1ull << 62;
    for (uint32_t i = 0; i < 32; ++i) {
        if (op >= result + one) { op -= result + one; result = (result >> 1) + one; }
        else result >>= 1;
        one >>= 2;
    }
    return (uint32_t)result;
}

static int bcuda_execute_residual_layer(BcudaGraphRuntime *runtime, const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer) {
    BcudaBuffer *lhs = 0, *rhs = 0, *output = 0;
    int status = bcuda_graph_table_buffer(model_buffer, model, layer->input0, &lhs);
    if (status == BC_STATUS_OK) status = bcuda_graph_table_buffer(model_buffer, model, layer->input1, &rhs);
    if (status == BC_STATUS_OK) status = bcuda_graph_table_buffer(model_buffer, model, layer->output, &output);
    if (status != BC_STATUS_OK) return status;
    if (lhs->dtype != BC_DTYPE_INT8 || rhs->dtype != BC_DTYPE_INT8 || output->dtype != BC_DTYPE_INT8 || lhs->size_bytes != rhs->size_bytes || lhs->size_bytes != output->size_bytes) return BC_STATUS_SHAPE_MISMATCH;
    uint32_t ls = layer->flags & BCUDA_RESIDUAL_LHS_SHIFT_MASK;
    uint32_t rs = (layer->flags & BCUDA_RESIDUAL_RHS_SHIFT_MASK) >> BCUDA_RESIDUAL_RHS_SHIFT_SHIFT;
    if (runtime != 0 && runtime->dma != 0 && runtime->dma->register_base != 0) {
        status = bcuda_axi_dma_submit_transformer_elementwise(runtime->dma, lhs, rhs, output, 1u, (uint32_t)output->size_bytes, 1u, ls, rs);
        return status == BC_STATUS_OK ? bcuda_axi_dma_poll_done(runtime->dma) : status;
    }
    int8_t *a = (int8_t *)lhs->virtual_address, *b = (int8_t *)rhs->virtual_address, *dst = (int8_t *)output->virtual_address;
    for (uint64_t i = 0; i < output->size_bytes; ++i) dst[i] = bcuda_saturate_int8((a[i] >> ls) + (b[i] >> rs));
    return BC_STATUS_OK;
}

static int bcuda_execute_gelu_layer(BcudaGraphRuntime *runtime, const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer) {
    BcudaBuffer *input = 0, *output = 0;
    int status = bcuda_graph_table_buffer(model_buffer, model, layer->input0, &input);
    if (status == BC_STATUS_OK) status = bcuda_graph_table_buffer(model_buffer, model, layer->output, &output);
    if (status != BC_STATUS_OK) return status;
    if (input->dtype != BC_DTYPE_INT8 || output->dtype != BC_DTYPE_INT8 || input->size_bytes != output->size_bytes) return BC_STATUS_SHAPE_MISMATCH;
    if (runtime != 0 && runtime->dma != 0 && runtime->dma->register_base != 0) {
        uint32_t columns = input->rank >= 2 && input->shape[1] ? input->shape[1] : (uint32_t)input->size_bytes;
        uint32_t rows = (uint32_t)(input->size_bytes / columns);
        status = bcuda_axi_dma_submit_transformer_elementwise(runtime->dma, input, 0, output, rows, columns, 0u, 0u, 0u);
        return status == BC_STATUS_OK ? bcuda_axi_dma_poll_done(runtime->dma) : status;
    }
    int8_t *src = (int8_t *)input->virtual_address, *dst = (int8_t *)output->virtual_address;
    for (uint64_t i = 0; i < input->size_bytes; ++i) {
        int32_t x = src[i], value;
        if (x <= -24) value = 0;
        else if (x >= 24) value = x;
        else { int32_t product = x * (x + 24); value = (product + (product >= 0 ? 24 : -24)) / 48; }
        dst[i] = (int8_t)value;
    }
    return BC_STATUS_OK;
}

static int bcuda_execute_layernorm_layer(BcudaGraphRuntime *runtime, const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer) {
    BcudaBuffer *input = 0, *output = 0, *gamma = 0, *beta = 0;
    int status = bcuda_graph_table_buffer(model_buffer, model, layer->input0, &input);
    if (status == BC_STATUS_OK) status = bcuda_graph_table_buffer(model_buffer, model, layer->output, &output);
    if (status == BC_STATUS_OK) status = bcuda_graph_table_buffer(model_buffer, model, layer->scale, &gamma);
    if (status == BC_STATUS_OK) status = bcuda_graph_table_buffer(model_buffer, model, layer->bias, &beta);
    if (status != BC_STATUS_OK) return status;
    uint64_t count = (uint64_t)layer->n * layer->c;
    int tiled_output = output->layout == BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8;
    uint64_t output_bytes = tiled_output ? (((uint64_t)layer->n + 7u) / 8u) * layer->c * 8u : count;
    if (layer->n == 0 || layer->c == 0 || input->dtype != BC_DTYPE_INT8 || input->layout != BC_LAYOUT_ROW_MAJOR || output->dtype != BC_DTYPE_INT8 || (output->layout != BC_LAYOUT_ROW_MAJOR && !tiled_output) || gamma->dtype != BC_DTYPE_INT16 || beta->dtype != BC_DTYPE_INT16 || input->size_bytes < count || output->size_bytes < output_bytes || gamma->size_bytes < 2ull * layer->c || beta->size_bytes < 2ull * layer->c) return BC_STATUS_SHAPE_MISMATCH;
    int8_t *src = (int8_t *)input->virtual_address, *dst = (int8_t *)output->virtual_address;
    int16_t *g = (int16_t *)gamma->virtual_address, *b = (int16_t *)beta->virtual_address;
    if(runtime!=0&&runtime->dma!=0&&runtime->dma->register_base!=0){status=bcuda_axi_dma_submit_transformer_rowwise(runtime->dma,input,gamma,beta,output,layer->n,layer->c,0u,layer->flags);return status==BC_STATUS_OK?bcuda_axi_dma_poll_done(runtime->dma):status;}
    if (tiled_output) memset(dst, 0, (size_t)output_bytes);
    for (uint32_t row = 0; row < layer->n; ++row) {
        int64_t sum = 0; uint64_t squares = 0;
        for (uint32_t i = 0; i < layer->c; ++i) { int32_t x = src[(uint64_t)row * layer->c + i]; sum += x; squares += (uint64_t)(x * x); }
        uint64_t variance = (uint64_t)layer->c * squares - (uint64_t)(sum * sum) + layer->flags;
        uint32_t denominator = bcuda_integer_sqrt(variance); if (denominator == 0) denominator = 1;
        for (uint32_t i = 0; i < layer->c; ++i) {
            int64_t centered = (int64_t)src[(uint64_t)row * layer->c + i] * layer->c - sum;
            int64_t norm_q14 = (centered << 14) / denominator;
            int64_t scaled_q8 = ((norm_q14 * g[i]) >> 14) + b[i];
            int64_t rounded = scaled_q8 + (scaled_q8 >= 0 ? 128 : -128);
            uint64_t dst_idx = tiled_output ? (((uint64_t)(row / 8u) * layer->c + i) * 8u + (row & 7u)) : (uint64_t)row * layer->c + i;
            dst[dst_idx] = bcuda_saturate_int8((int32_t)(rounded >> 8));
        }
    }
    return BC_STATUS_OK;
}

static int bcuda_execute_softmax_layer(BcudaGraphRuntime *runtime, const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer) {
    static const uint16_t lut[17] = {32767,25520,19874,15480,12055,9388,7312,5695,4435,3454,2690,2095,1631,1270,989,770,600};
    BcudaBuffer *input = 0, *output = 0;
    int status = bcuda_graph_table_buffer(model_buffer, model, layer->input0, &input);
    if (status == BC_STATUS_OK) status = bcuda_graph_table_buffer(model_buffer, model, layer->output, &output);
    if (status != BC_STATUS_OK) return status;
    uint64_t count = (uint64_t)layer->n * layer->c;
    int tiled_output = output->layout == BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8;
    uint64_t output_bytes = tiled_output ? ((uint64_t)(layer->n + 7u) / 8u) * layer->c * 8u : count;
    if (layer->n == 0 || layer->c == 0 || layer->c > 256 || input->dtype != BC_DTYPE_INT8 ||
        input->layout != BC_LAYOUT_ROW_MAJOR || output->dtype != BC_DTYPE_INT8 ||
        (output->layout != BC_LAYOUT_ROW_MAJOR && !tiled_output) ||
        input->size_bytes < count || output->size_bytes < output_bytes) return BC_STATUS_SHAPE_MISMATCH;
    if(runtime!=0&&runtime->dma!=0&&runtime->dma->register_base!=0){status=bcuda_axi_dma_submit_transformer_rowwise(runtime->dma,input,0,0,output,layer->n,layer->c,1u,0u);return status==BC_STATUS_OK?bcuda_axi_dma_poll_done(runtime->dma):status;}
    int8_t *src = (int8_t *)input->virtual_address; uint8_t *dst = (uint8_t *)output->virtual_address; uint16_t exps[256];
    for (uint32_t row = 0; row < layer->n; ++row) {
        int32_t maximum = -128; uint32_t total = 0;
        for (uint32_t i = 0; i < layer->c; ++i) if (src[(uint64_t)row * layer->c + i] > maximum) maximum = src[(uint64_t)row * layer->c + i];
        for (uint32_t i = 0; i < layer->c; ++i) { uint32_t d = maximum - src[(uint64_t)row * layer->c + i]; exps[i] = d <= 16 ? lut[d] : 0; total += exps[i]; }
        for (uint32_t i = 0; i < layer->c; ++i) {
            uint64_t destination = tiled_output
                ? ((uint64_t)(row / 8u) * layer->c + i) * 8u + (row & 7u)
                : (uint64_t)row * layer->c + i;
            dst[destination] = total == 0 ? 0 : (uint8_t)((exps[i] * 127u + (total >> 1)) / total);
        }
    }
    return BC_STATUS_OK;
}

static int bcuda_execute_upsample_layer(BcudaGraphRuntime *runtime, const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer) {
    BcudaBuffer *input = 0;
    BcudaBuffer *output = 0;
    int status = bcuda_graph_table_buffer(model_buffer, model, layer->input0, &input);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_graph_table_buffer(model_buffer, model, layer->output, &output);
    if (status != BC_STATUS_OK) {
        return status;
    }
    if (input->dtype != output->dtype) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    uint32_t elem_bytes = bcuda_dtype_bytes(input->dtype);
    if (elem_bytes == 0) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    if (layer->n == 0 || layer->c == 0 || layer->h == 0 || layer->w == 0 || layer->out_h == 0 || layer->out_w == 0) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    uint64_t input_bytes = (uint64_t)layer->n * layer->c * layer->h * layer->w * elem_bytes;
    uint64_t output_bytes = (uint64_t)layer->n * layer->c * layer->out_h * layer->out_w * elem_bytes;
    if (input_bytes > input->size_bytes || output_bytes > output->size_bytes) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    if (runtime != 0 && runtime->dma != 0 && runtime->dma->register_base != 0) {
        status = bcuda_axi_dma_submit_upsample(runtime->dma, layer, input, output);
        return status == BC_STATUS_OK ? bcuda_axi_dma_poll_done(runtime->dma) : status;
    }

    const uint8_t *src = (const uint8_t *)input->virtual_address;
    uint8_t *dst = (uint8_t *)output->virtual_address;
    for (uint32_t n = 0; n < layer->n; n++) {
        for (uint32_t c = 0; c < layer->c; c++) {
            for (uint32_t oh = 0; oh < layer->out_h; oh++) {
                uint32_t ih = (uint32_t)(((uint64_t)oh * layer->h) / layer->out_h);
                for (uint32_t ow = 0; ow < layer->out_w; ow++) {
                    uint32_t iw = (uint32_t)(((uint64_t)ow * layer->w) / layer->out_w);
                    uint64_t src_idx = (((uint64_t)n * layer->c + c) * layer->h + ih) * layer->w + iw;
                    uint64_t dst_idx = (((uint64_t)n * layer->c + c) * layer->out_h + oh) * layer->out_w + ow;
                    memcpy(dst + dst_idx * elem_bytes, src + src_idx * elem_bytes, elem_bytes);
                }
            }
        }
    }
    return BC_STATUS_OK;
}

static int bcuda_concat_copy_input(const BcudaBuffer *input, BcudaBuffer *output, uint32_t dtype, uint32_t n_batches, uint32_t in_channels, uint32_t out_channels, uint32_t h, uint32_t w, uint32_t channel_offset) {
    uint32_t elem_bytes = bcuda_dtype_bytes(dtype);
    if (input == 0 || input->dtype != dtype || input->size_bytes < (uint64_t)n_batches * in_channels * h * w * elem_bytes) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    const uint8_t *src = (const uint8_t *)input->virtual_address;
    uint8_t *dst = (uint8_t *)output->virtual_address;
    for (uint32_t n = 0; n < n_batches; n++) {
        for (uint32_t c = 0; c < in_channels; c++) {
            uint64_t src_idx = ((uint64_t)n * in_channels + c) * h * w;
            uint64_t dst_idx = ((uint64_t)n * out_channels + channel_offset + c) * h * w;
            memcpy(dst + dst_idx * elem_bytes, src + src_idx * elem_bytes, (size_t)h * w * elem_bytes);
        }
    }
    return BC_STATUS_OK;
}

static int bcuda_execute_concat_layer(BcudaGraphRuntime *runtime, const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer) {
    BcudaBuffer *inputs[3] = {0, 0, 0};
    BcudaBuffer *output = 0;
    uint32_t input_indexes[3] = {layer->input0, layer->input1, layer->input2};
    uint32_t input_count = layer->input2 != 0 ? 3u : (layer->input1 != 0 ? 2u : 1u);
    int status = bcuda_graph_table_buffer(model_buffer, model, layer->output, &output);
    if (status != BC_STATUS_OK) {
        return status;
    }
    if (layer->n == 0 || layer->h == 0 || layer->w == 0 || layer->out_c == 0) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    uint32_t elem_bytes = bcuda_dtype_bytes(output->dtype);
    if (elem_bytes == 0) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    uint64_t output_bytes = (uint64_t)layer->n * layer->out_c * layer->h * layer->w * elem_bytes;
    if (output_bytes > output->size_bytes) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    uint32_t channel_offset = 0;
    for (uint32_t i = 0; i < input_count; i++) {
        status = bcuda_graph_table_buffer(model_buffer, model, input_indexes[i], &inputs[i]);
        if (status != BC_STATUS_OK) {
            return status;
        }
        if (inputs[i]->dtype != output->dtype) {
            return BC_STATUS_SHAPE_MISMATCH;
        }
        uint64_t denom = (uint64_t)layer->n * layer->h * layer->w * elem_bytes;
        if (denom == 0 || (inputs[i]->size_bytes % denom) != 0) {
            return BC_STATUS_SHAPE_MISMATCH;
        }
        uint32_t in_channels = (uint32_t)(inputs[i]->size_bytes / denom);
        if (in_channels == 0 || channel_offset + in_channels > layer->out_c) {
            return BC_STATUS_SHAPE_MISMATCH;
        }
        if (runtime == 0 || runtime->dma == 0 || runtime->dma->register_base == 0) {
            status = bcuda_concat_copy_input(inputs[i], output, output->dtype, layer->n, in_channels, layer->out_c, layer->h, layer->w, channel_offset);
            if (status != BC_STATUS_OK) {
                return status;
            }
        }
        channel_offset += in_channels;
    }
    if (channel_offset != layer->out_c) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    if (runtime != 0 && runtime->dma != 0 && runtime->dma->register_base != 0) {
        status = bcuda_axi_dma_submit_concat(runtime->dma, layer, inputs[0], inputs[1], input_count == 3 ? inputs[2] : 0, output);
        return status == BC_STATUS_OK ? bcuda_axi_dma_poll_done(runtime->dma) : status;
    }
    return BC_STATUS_OK;
}

static int bcuda_execute_pool_layer(BcudaGraphRuntime *runtime, const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer) {
    BcudaBuffer *input = 0;
    BcudaBuffer *output = 0;
    int status = bcuda_graph_table_buffer(model_buffer, model, layer->input0, &input);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_graph_table_buffer(model_buffer, model, layer->output, &output);
    if (status != BC_STATUS_OK) {
        return status;
    }
    if (input->dtype != BC_DTYPE_INT8 || output->dtype != BC_DTYPE_INT8) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    if (layer->n == 0 || layer->c == 0 || layer->h == 0 || layer->w == 0 || layer->out_h == 0 || layer->out_w == 0 ||
        layer->kernel_h == 0 || layer->kernel_w == 0 || layer->stride_h == 0 || layer->stride_w == 0) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    uint64_t input_elements = (uint64_t)layer->n * layer->c * layer->h * layer->w;
    uint64_t output_elements = (uint64_t)layer->n * layer->c * layer->out_h * layer->out_w;
    if (input_elements > input->size_bytes || output_elements > output->size_bytes) {
        return BC_STATUS_SHAPE_MISMATCH;
    }
    if (runtime != 0 && runtime->dma != 0 && runtime->dma->register_base != 0) {
        status = (layer->flags & BCUDA_POOL_AVERAGE) != 0u
            ? bcuda_axi_dma_submit_avgpool(runtime->dma, layer, input, output)
            : bcuda_axi_dma_submit_maxpool(runtime->dma, layer, input, output);
        return status == BC_STATUS_OK ? bcuda_axi_dma_poll_done(runtime->dma) : status;
    }

    const int8_t *src = (const int8_t *)input->virtual_address;
    int8_t *dst = (int8_t *)output->virtual_address;
    for (uint32_t n = 0; n < layer->n; n++) {
        for (uint32_t c = 0; c < layer->c; c++) {
            for (uint32_t oh = 0; oh < layer->out_h; oh++) {
                for (uint32_t ow = 0; ow < layer->out_w; ow++) {
                    int32_t sum = 0;
                    uint32_t count = 0;
                    int8_t best = -128;
                    for (uint32_t kh = 0; kh < layer->kernel_h; kh++) {
                        int32_t ih = (int32_t)(oh * layer->stride_h + kh) - (int32_t)layer->pad_h;
                        if (ih < 0 || ih >= (int32_t)layer->h) {
                            continue;
                        }
                        for (uint32_t kw = 0; kw < layer->kernel_w; kw++) {
                            int32_t iw = (int32_t)(ow * layer->stride_w + kw) - (int32_t)layer->pad_w;
                            if (iw < 0 || iw >= (int32_t)layer->w) {
                                continue;
                            }
                            uint64_t src_idx = (((uint64_t)n * layer->c + c) * layer->h + (uint32_t)ih) * layer->w + (uint32_t)iw;
                            sum += src[src_idx];
                            count++;
                            if (src[src_idx] > best) {
                                best = src[src_idx];
                            }
                        }
                    }
                    uint64_t dst_idx = (((uint64_t)n * layer->c + c) * layer->out_h + oh) * layer->out_w + ow;
                    dst[dst_idx] = (layer->flags & BCUDA_POOL_AVERAGE) != 0u
                        ? (int8_t)(sum < 0 ? -((-sum) / (int32_t)count) : sum / (int32_t)count)
                        : best;
                }
            }
        }
    }
    return BC_STATUS_OK;
}

static int bcuda_execute_conv2d_layer(BcudaGraphRuntime *runtime, const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer) {
    BcudaBuffer *input = 0;
    BcudaBuffer *weight = 0;
    BcudaBuffer *output = 0;
    BcudaBuffer *bias = 0;
    int status = bcuda_graph_table_buffer(model_buffer, model, layer->input0, &input);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_graph_table_buffer(model_buffer, model, layer->weight, &weight);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_graph_table_buffer(model_buffer, model, layer->output, &output);
    if (status != BC_STATUS_OK) {
        return status;
    }
    if (layer->bias != 0) {
        status = bcuda_graph_table_buffer(model_buffer, model, layer->bias, &bias);
        if (status != BC_STATUS_OK) {
            return status;
        }
    }
    if (runtime != 0 && runtime->dma != 0 && runtime->dma->register_base != 0) {
        status = bcuda_axi_dma_submit_conv2d(runtime->dma, layer, input, weight, bias, output);
        return status == BC_STATUS_OK ? bcuda_axi_dma_poll_done(runtime->dma) : status;
    }
    if (input->dtype != BC_DTYPE_INT8 || weight->dtype != BC_DTYPE_INT8 || output->dtype != BC_DTYPE_INT32 ||
        (bias != 0 && bias->dtype != BC_DTYPE_INT32)) {
        return BC_STATUS_UNSUPPORTED_DTYPE;
    }
    if (layer->groups != 1 || layer->n == 0 || layer->c == 0 || layer->h == 0 || layer->w == 0 || layer->out_c == 0 ||
        layer->out_h == 0 || layer->out_w == 0 || layer->kernel_h == 0 || layer->kernel_w == 0 ||
        layer->stride_h == 0 || layer->stride_w == 0) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    uint64_t input_elements = (uint64_t)layer->n * layer->c * layer->h * layer->w;
    uint64_t reduction = (uint64_t)layer->c * layer->kernel_h * layer->kernel_w;
    uint64_t weight_elements = (layer->flags & BCUDA_CONV_PRESLICED_WEIGHTS) != 0u
        ? ((layer->out_c + 15u) / 16u) * ((reduction + 15u) / 16u) * 256u
        : (uint64_t)layer->out_c * reduction;
    uint64_t output_bytes = (uint64_t)layer->n * layer->out_c * layer->out_h * layer->out_w * sizeof(int32_t);
    if (input_elements > input->size_bytes || weight_elements > weight->size_bytes || output_bytes > output->size_bytes ||
        (bias != 0 && (uint64_t)layer->out_c * sizeof(int32_t) > bias->size_bytes)) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    const int8_t *src = (const int8_t *)input->virtual_address;
    const int8_t *wgt = (const int8_t *)weight->virtual_address;
    const uint8_t *bias_data = bias == 0 ? 0 : (const uint8_t *)bias->virtual_address;
    uint8_t *dst = (uint8_t *)output->virtual_address;
    for (uint32_t n = 0; n < layer->n; n++) {
        for (uint32_t oc = 0; oc < layer->out_c; oc++) {
            for (uint32_t oh = 0; oh < layer->out_h; oh++) {
                for (uint32_t ow = 0; ow < layer->out_w; ow++) {
                    int32_t acc = bias_data == 0 ? 0 : bcuda_read_i32(bias_data + oc * sizeof(int32_t));
                    for (uint32_t ic = 0; ic < layer->c; ic++) {
                        for (uint32_t kh = 0; kh < layer->kernel_h; kh++) {
                            int32_t ih = (int32_t)(oh * layer->stride_h + kh) - (int32_t)layer->pad_h;
                            if (ih < 0 || ih >= (int32_t)layer->h) {
                                continue;
                            }
                            for (uint32_t kw = 0; kw < layer->kernel_w; kw++) {
                                int32_t iw = (int32_t)(ow * layer->stride_w + kw) - (int32_t)layer->pad_w;
                                if (iw < 0 || iw >= (int32_t)layer->w) {
                                    continue;
                                }
                                uint64_t input_idx = (((uint64_t)n * layer->c + ic) * layer->h + (uint32_t)ih) * layer->w + (uint32_t)iw;
                                uint64_t k = ((uint64_t)ic * layer->kernel_h + kh) * layer->kernel_w + kw;
                                uint64_t weight_idx = (layer->flags & BCUDA_CONV_PRESLICED_WEIGHTS) != 0u
                                    ? ((((uint64_t)oc / 16u) * ((reduction + 15u) / 16u) + k / 16u) * 256u) + (k & 15u) * 16u + (oc & 15u)
                                    : (uint64_t)oc * reduction + k;
                                acc += (int32_t)src[input_idx] * (int32_t)wgt[weight_idx];
                            }
                        }
                    }
                    uint64_t output_idx = (((uint64_t)n * layer->out_c + oc) * layer->out_h + oh) * layer->out_w + ow;
                    bcuda_write_i32(dst + output_idx * sizeof(int32_t), acc);
                }
            }
        }
    }
    return BC_STATUS_OK;
}

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

    if (desc->A_layout == BC_LAYOUT_TRANSFORMER_ACTIVATION_TILED_INT8 &&
        (desc->B_layout == BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT8 ||
         desc->B_layout == BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT4 ||
         desc->B_layout == BC_LAYOUT_TRANSFORMER_DYNAMIC_B_TILED_INT8) &&
        desc->C_layout == BC_LAYOUT_ROW_MAJOR) {
        if (A->layout != desc->A_layout || B->layout != desc->B_layout || C->layout != desc->C_layout) {
            return BC_STATUS_SHAPE_MISMATCH;
        }
        status = bcuda_axi_dma_submit_transformer_gemm(
            runtime->dma, desc, A, B, C,
            desc->B_layout == BC_LAYOUT_TRANSFORMER_WEIGHT_TILED_INT4);
        return status == BC_STATUS_OK ? bcuda_axi_dma_poll_done(runtime->dma) : status;
    }

    if ((desc->M & 15u) != 0u || (desc->N & 15u) != 0u || (desc->K & 15u) != 0u) {
        return bcuda_execute_padded_gemm(runtime, desc, A, B, C);
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

    BcudaModelDesc model_desc;
    status = bcuda_copy_model_desc(model, &model_desc);
    if (status != BC_STATUS_OK) {
        return status;
    }
    if (model_desc.input_buffer >= model_desc.num_buffers || model_desc.output_buffer >= model_desc.num_buffers) {
        return BC_STATUS_INVALID_HANDLE;
    }

    uint32_t model_input_id = 0;
    uint32_t model_output_id = 0;
    status = bcuda_graph_table_buffer_id(model, &model_desc, model_desc.input_buffer, &model_input_id);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_graph_table_buffer_id(model, &model_desc, model_desc.output_buffer, &model_output_id);
    if (status != BC_STATUS_OK) {
        return status;
    }
    if (model_input_id != input->buffer_id || model_output_id != output->buffer_id) {
        return BC_STATUS_INVALID_HANDLE;
    }

    BcudaBuffer *diagnostics_buffer = 0;
    BcudaDiagnosticsHeader *diagnostics_header = 0;
    BcudaDiagnosticsRecord *diagnostics_records = 0;
    status = bcuda_diag_open(
        desc,
        model_desc.num_layers,
        &diagnostics_buffer,
        &diagnostics_header,
        &diagnostics_records
    );
    if (status != BC_STATUS_OK) {
        return status;
    }
    (void)diagnostics_buffer;
    uint64_t graph_started_ns = diagnostics_header == 0 ? 0 : bcuda_now_ns();
    if (diagnostics_header != 0 && runtime->dma != 0 && runtime->dma->register_base != 0) {
        /* Start each report from a clean max/completed-op baseline without
         * enabling single-step mode or freezing the free-running counters. */
        bcuda_axi_dma_debug_control(runtime->dma, BCUDA_DEBUG_CLEAR);
        bcuda_axi_dma_debug_control(runtime->dma, 0u);
    }

    for (uint32_t i = 0; i < model_desc.num_layers; i++) {
        BcudaLayerDesc layer;
        status = bcuda_copy_layer(model, &model_desc, i, &layer);
        if (status != BC_STATUS_OK) {
            return status;
        }
        BcudaDiagnosticsRecord *record = diagnostics_header == 0 ? 0 : &diagnostics_records[i];
        if (record != 0) {
            memset(record, 0, sizeof(*record));
            record->layer_index = i;
            record->op_type = layer.op_type;
            if ((desc->flags & BCUDA_DIAG_MEMORY) != 0) {
                bcuda_diag_resources(model, &model_desc, &layer, record);
            }
            if ((desc->flags & BCUDA_DIAG_OPERATIONS) != 0) {
                record->operations = bcuda_diag_operations(model, &model_desc, &layer);
            }
        }
        uint64_t layer_started_ns = record == 0 ? 0 : bcuda_now_ns();
        BcudaTransformerDebugSnapshot debug_before;
        uint64_t gemm_bias_elapsed_ns = 0;
        int debug_snapshot_valid = record != 0 && runtime->dma != 0 && runtime->dma->register_base != 0 &&
            (layer.op_type == BC_OP_GEMM || layer.op_type == BC_OP_REQUANTIZE ||
             layer.op_type >= BC_OP_LAYERNORM) &&
            bcuda_axi_dma_read_transformer_debug(runtime->dma, &debug_before) == BC_STATUS_OK;
        switch (layer.op_type) {
            case BC_OP_GEMM:
                status = bcuda_execute_gemm_layer(
                    runtime, model, &model_desc, &layer, &gemm_bias_elapsed_ns);
                break;
            case BC_OP_CONV2D_GEMM:
                status = bcuda_execute_conv2d_layer(runtime, model, &model_desc, &layer);
                break;
            case BC_OP_ACTIVATION:
                status = bcuda_execute_activation_layer(runtime, model, &model_desc, &layer);
                break;
            case BC_OP_REQUANTIZE:
                status = bcuda_execute_requantize_layer(runtime, model, &model_desc, &layer);
                break;
            case BC_OP_POOL:
                status = bcuda_execute_pool_layer(runtime, model, &model_desc, &layer);
                break;
            case BC_OP_UPSAMPLE:
                status = bcuda_execute_upsample_layer(runtime, model, &model_desc, &layer);
                break;
            case BC_OP_CONCAT:
                status = bcuda_execute_concat_layer(runtime, model, &model_desc, &layer);
                break;
            case BC_OP_RESHAPE:
                status = bcuda_execute_reshape_layer(model, &model_desc, &layer);
                break;
            case BC_OP_LAYERNORM:
                status = bcuda_execute_layernorm_layer(runtime, model, &model_desc, &layer);
                break;
            case BC_OP_GELU:
                status = bcuda_execute_gelu_layer(runtime, model, &model_desc, &layer);
                break;
            case BC_OP_SOFTMAX:
                status = bcuda_execute_softmax_layer(runtime, model, &model_desc, &layer);
                break;
            case BC_OP_RESIDUAL_ADD:
                status = bcuda_execute_residual_layer(runtime, model, &model_desc, &layer);
                break;
            case BC_OP_ACTIVATION_B_PACK:
                status = bcuda_execute_activation_b_pack_layer(runtime, model, &model_desc, &layer);
                break;
            case BC_OP_TOKEN_CONCAT:
                status = bcuda_execute_token_concat_layer(model, &model_desc, &layer);
                break;
            case BC_OP_TOKEN_EMBEDDING:
                status = bcuda_execute_token_embedding_layer(model, &model_desc, &layer);
                break;
            default:
                status = BC_STATUS_INVALID_COMMAND;
                break;
        }
        if (record != 0) {
            uint64_t finished_ns = bcuda_now_ns();
            record->status = (uint32_t)status;
            record->gemm_bias_elapsed_ns = gemm_bias_elapsed_ns > UINT32_MAX
                ? UINT32_MAX : (uint32_t)gemm_bias_elapsed_ns;
            if ((desc->flags & BCUDA_DIAG_TIMING) != 0 && finished_ns >= layer_started_ns) {
                record->elapsed_ns = finished_ns - layer_started_ns;
                record->cycles = record->elapsed_ns * diagnostics_header->clock_hz / 1000000000ull;
            }
            if ((desc->flags & BCUDA_DIAG_OPERATIONS) != 0) {
                record->throughput_ops_per_second = bcuda_diag_rate(record->operations, record->elapsed_ns);
            }
            if ((desc->flags & BCUDA_DIAG_UTILIZATION) != 0) {
                record->utilization_ppm = bcuda_diag_utilization(
                    record->operations,
                    record->cycles,
                    diagnostics_header->peak_operations_per_cycle
                );
            }
            if (debug_snapshot_valid) {
                BcudaTransformerDebugSnapshot after;
                if (bcuda_axi_dma_read_transformer_debug(runtime->dma, &after) == BC_STATUS_OK) {
                    record->hardware_total_cycles = after.total_cycles - debug_before.total_cycles;
                    record->hardware_busy_cycles = after.busy_cycles - debug_before.busy_cycles;
                    record->hardware_memory_stall_cycles = after.memory_stall_cycles - debug_before.memory_stall_cycles;
                    record->hardware_input_stall_cycles = after.input_stall_cycles - debug_before.input_stall_cycles;
                    record->hardware_output_stall_cycles = after.output_stall_cycles - debug_before.output_stall_cycles;
                    record->hardware_last_op_cycles = after.last_op_cycles;
                    record->hardware_max_op_cycles = after.max_op_cycles;
                    record->hardware_gemm_load_a_cycles = after.gemm_load_a_cycles - debug_before.gemm_load_a_cycles;
                    record->hardware_gemm_load_b_cycles = after.gemm_load_b_cycles - debug_before.gemm_load_b_cycles;
                    record->hardware_gemm_compute_cycles = after.gemm_compute_cycles - debug_before.gemm_compute_cycles;
                    record->hardware_gemm_write_cycles = after.gemm_write_cycles - debug_before.gemm_write_cycles;
                    record->hardware_completed_ops = after.completed_ops - debug_before.completed_ops;
                    record->hardware_status = (uint32_t)after.active_op |
                        ((uint32_t)after.error_latched << 8) |
                        ((uint32_t)after.trace_valid << 9) |
                        ((uint32_t)after.trace_event << 16);
                }
            }
            diagnostics_header->layer_count = i + 1;
        }
        if (status != BC_STATUS_OK) {
            if (diagnostics_header != 0) {
                uint64_t finished_ns = bcuda_now_ns();
                diagnostics_header->total_elapsed_ns =
                    (desc->flags & BCUDA_DIAG_TIMING) != 0 && finished_ns >= graph_started_ns
                    ? finished_ns - graph_started_ns : 0;
                diagnostics_header->arena_bytes_used =
                    (desc->flags & BCUDA_DIAG_MEMORY) != 0 ? bcuda_buffer_bytes_used() : 0;
            }
            return status;
        }
    }
    if (diagnostics_header != 0) {
        uint64_t finished_ns = bcuda_now_ns();
        diagnostics_header->total_elapsed_ns =
            (desc->flags & BCUDA_DIAG_TIMING) != 0 && finished_ns >= graph_started_ns
            ? finished_ns - graph_started_ns : 0;
        diagnostics_header->arena_bytes_used =
            (desc->flags & BCUDA_DIAG_MEMORY) != 0 ? bcuda_buffer_bytes_used() : 0;
    }
    return BC_STATUS_OK;
}
