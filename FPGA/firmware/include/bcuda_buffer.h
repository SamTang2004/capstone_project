#ifndef BCUDA_BUFFER_H
#define BCUDA_BUFFER_H

#include <stddef.h>
#include <stdint.h>

#include "bcuda_protocol.h"

#define BCUDA_MAX_BUFFERS 512

typedef int (*BcudaBufferSyncFn)(uint64_t offset, uint64_t bytes, void *context);

typedef struct {
    uint32_t buffer_id;
    uintptr_t physical_address;
    void *virtual_address;
    uint64_t size_bytes;
    uint32_t dtype;
    uint32_t layout;
    uint32_t usage;
    uint32_t rank;
    uint32_t shape[4];
    uint32_t state;
} BcudaBuffer;

void bcuda_buffer_manager_init(uint8_t *arena, uint64_t arena_bytes);
void bcuda_buffer_manager_init_with_physical(uint8_t *arena, uintptr_t physical_base, uint64_t arena_bytes);
void bcuda_buffer_manager_set_sync_hooks(
    BcudaBufferSyncFn to_device,
    BcudaBufferSyncFn from_device,
    void *context
);
int bcuda_alloc_buffer(uint64_t size_bytes, BcudaTensorDesc desc, uint32_t usage, uint32_t *out_id);
int bcuda_free_buffer(uint32_t buffer_id);
int bcuda_get_buffer(uint32_t buffer_id, BcudaBuffer **out);
int bcuda_write_buffer(uint32_t buffer_id, uint64_t offset, const void *src, uint64_t bytes);
int bcuda_read_buffer(uint32_t buffer_id, uint64_t offset, void *dst, uint64_t bytes);
uint64_t bcuda_buffer_bytes_used(void);

#endif
