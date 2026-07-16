#include "bcuda_graph.h"

#include <stdint.h>
#include <string.h>

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

static int bcuda_execute_gemm_layer(BcudaGraphRuntime *runtime, const BcudaBuffer *model_buffer, const BcudaModelDesc *model, const BcudaLayerDesc *layer) {
    uint32_t input_id = 0;
    uint32_t weight_id = 0;
    uint32_t output_id = 0;
    int status = bcuda_graph_table_buffer_id(model_buffer, model, layer->input0, &input_id);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_graph_table_buffer_id(model_buffer, model, layer->weight, &weight_id);
    if (status != BC_STATUS_OK) {
        return status;
    }
    status = bcuda_graph_table_buffer_id(model_buffer, model, layer->output, &output_id);
    if (status != BC_STATUS_OK) {
        return status;
    }

    BcudaGemmDesc gemm;
    memset(&gemm, 0, sizeof(gemm));
    gemm.A_buffer = input_id;
    gemm.B_buffer = weight_id;
    gemm.C_buffer = output_id;
    gemm.M = layer->n;
    gemm.N = layer->out_c;
    gemm.K = layer->c;
    gemm.A_dtype = BC_DTYPE_INT8;
    gemm.B_dtype = BC_DTYPE_INT8;
    gemm.C_dtype = BC_DTYPE_INT32;
    gemm.A_layout = BC_LAYOUT_ROW_MAJOR;
    gemm.B_layout = BC_LAYOUT_ROW_MAJOR;
    gemm.C_layout = BC_LAYOUT_ROW_MAJOR;
    gemm.flags = layer->flags;
    return bcuda_execute_gemm(runtime, &gemm);
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
        scale->size_bytes < sizeof(BcudaRequantParams) || input->size_bytes / sizeof(int32_t) > output->size_bytes) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    BcudaRequantParams params;
    memcpy(&params, scale->virtual_address, sizeof(params));
    uint32_t elements = (uint32_t)(input->size_bytes / sizeof(int32_t));
    if (elements == 0 || params.right_shift > 31u) {
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
        status = bcuda_axi_dma_submit_requantize(runtime->dma, &desc, input, output);
        return status == BC_STATUS_OK ? bcuda_axi_dma_poll_done(runtime->dma) : status;
    }

    const uint8_t *src = (const uint8_t *)input->virtual_address;
    int8_t *dst = (int8_t *)output->virtual_address;
    for (uint32_t i = 0; i < elements; i++) {
        dst[i] = bcuda_requantize_value(bcuda_read_i32(src + i * sizeof(int32_t)), &params);
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
    uint64_t weight_elements = (uint64_t)layer->out_c * layer->c * layer->kernel_h * layer->kernel_w;
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
                                uint64_t weight_idx = (((uint64_t)oc * layer->c + ic) * layer->kernel_h + kh) * layer->kernel_w + kw;
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

    for (uint32_t i = 0; i < model_desc.num_layers; i++) {
        BcudaLayerDesc layer;
        status = bcuda_copy_layer(model, &model_desc, i, &layer);
        if (status != BC_STATUS_OK) {
            return status;
        }
        switch (layer.op_type) {
            case BC_OP_GEMM:
                status = bcuda_execute_gemm_layer(runtime, model, &model_desc, &layer);
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
            default:
                status = BC_STATUS_INVALID_COMMAND;
                break;
        }
        if (status != BC_STATUS_OK) {
            return status;
        }
    }
    return BC_STATUS_OK;
}
