#include "bcuda_buffer.h"

#include <string.h>

#include "bcuda_status.h"

#define BUFFER_STATE_FREE 0u
#define BUFFER_STATE_USED 1u

static uint8_t *g_arena;
static uint64_t g_arena_bytes;
static uint64_t g_arena_offset;
static uint32_t g_next_buffer_id;
static BcudaBuffer g_buffers[BCUDA_MAX_BUFFERS];

static uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

void bcuda_buffer_manager_init(uint8_t *arena, uint64_t arena_bytes) {
    g_arena = arena;
    g_arena_bytes = arena_bytes;
    g_arena_offset = 0;
    g_next_buffer_id = 1;
    memset(g_buffers, 0, sizeof(g_buffers));
}

int bcuda_alloc_buffer(uint64_t size_bytes, BcudaTensorDesc desc, uint32_t usage, uint32_t *out_id) {
    if (g_arena == 0 || out_id == 0 || size_bytes == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    if (desc.rank > 4) {
        return BC_STATUS_SHAPE_MISMATCH;
    }

    uint32_t slot = BCUDA_MAX_BUFFERS;
    for (uint32_t i = 0; i < BCUDA_MAX_BUFFERS; i++) {
        if (g_buffers[i].state == BUFFER_STATE_FREE) {
            slot = i;
            break;
        }
    }
    if (slot == BCUDA_MAX_BUFFERS) {
        return BC_STATUS_OUT_OF_MEMORY;
    }

    uint64_t aligned_offset = align_up_u64(g_arena_offset, 64);
    if (aligned_offset + size_bytes > g_arena_bytes) {
        return BC_STATUS_OUT_OF_MEMORY;
    }

    BcudaBuffer *buffer = &g_buffers[slot];
    buffer->buffer_id = g_next_buffer_id++;
    buffer->physical_address = (uintptr_t)(g_arena + aligned_offset);
    buffer->virtual_address = g_arena + aligned_offset;
    buffer->size_bytes = size_bytes;
    buffer->dtype = desc.dtype;
    buffer->layout = desc.layout;
    buffer->usage = usage;
    buffer->rank = desc.rank;
    buffer->shape[0] = desc.dim0;
    buffer->shape[1] = desc.dim1;
    buffer->shape[2] = desc.dim2;
    buffer->shape[3] = desc.dim3;
    buffer->state = BUFFER_STATE_USED;

    g_arena_offset = aligned_offset + size_bytes;
    *out_id = buffer->buffer_id;
    return BC_STATUS_OK;
}

int bcuda_free_buffer(uint32_t buffer_id) {
    BcudaBuffer *buffer = 0;
    int status = bcuda_get_buffer(buffer_id, &buffer);
    if (status != BC_STATUS_OK) {
        return status;
    }
    memset(buffer, 0, sizeof(*buffer));
    return BC_STATUS_OK;
}

int bcuda_get_buffer(uint32_t buffer_id, BcudaBuffer **out) {
    if (out == 0) {
        return BC_STATUS_GENERAL_ERROR;
    }
    for (uint32_t i = 0; i < BCUDA_MAX_BUFFERS; i++) {
        if (g_buffers[i].state == BUFFER_STATE_USED && g_buffers[i].buffer_id == buffer_id) {
            *out = &g_buffers[i];
            return BC_STATUS_OK;
        }
    }
    *out = 0;
    return BC_STATUS_INVALID_HANDLE;
}

int bcuda_write_buffer(uint32_t buffer_id, uint64_t offset, const void *src, uint64_t bytes) {
    BcudaBuffer *buffer = 0;
    int status = bcuda_get_buffer(buffer_id, &buffer);
    if (status != BC_STATUS_OK) {
        return status;
    }
    if (src == 0 || offset + bytes > buffer->size_bytes) {
        return BC_STATUS_GENERAL_ERROR;
    }
    memcpy((uint8_t *)buffer->virtual_address + offset, src, (size_t)bytes);
    return BC_STATUS_OK;
}

int bcuda_read_buffer(uint32_t buffer_id, uint64_t offset, void *dst, uint64_t bytes) {
    BcudaBuffer *buffer = 0;
    int status = bcuda_get_buffer(buffer_id, &buffer);
    if (status != BC_STATUS_OK) {
        return status;
    }
    if (dst == 0 || offset + bytes > buffer->size_bytes) {
        return BC_STATUS_GENERAL_ERROR;
    }
    memcpy(dst, (const uint8_t *)buffer->virtual_address + offset, (size_t)bytes);
    return BC_STATUS_OK;
}

uint64_t bcuda_buffer_bytes_used(void) {
    return g_arena_offset;
}
