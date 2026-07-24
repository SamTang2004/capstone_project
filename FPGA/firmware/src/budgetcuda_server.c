#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "bcuda_axi_dma.h"
#include "bcuda_buffer.h"
#include "bcuda_graph.h"
#include "bcuda_protocol_server.h"

#define BCUDA_XLNK_DEVICE "/dev/xlnk"
#define BCUDA_MEM_DEVICE "/dev/mem"
#define BCUDA_XLNK_IOCALLOCBUF 0xC0045802u
#define BCUDA_MMIO_BYTES 0x10000u
#define BCUDA_DEFAULT_ARENA_MIB 256u

typedef struct {
    int fd;
    int buffer_id;
    uint32_t physical_address;
    size_t bytes;
    uint8_t *virtual_address;
} BcudaCmaArena;

static size_t align_page(size_t value) {
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    return (value + page - 1u) & ~(page - 1u);
}

static int cma_alloc(BcudaCmaArena *arena, size_t bytes) {
    uint8_t request[128];
    uint32_t requested = (uint32_t)align_page(bytes);
    memset(arena, 0, sizeof(*arena));
    arena->fd = -1;
    memset(request, 0, sizeof(request));
    memcpy(request, &requested, sizeof(requested));
    arena->fd = open(BCUDA_XLNK_DEVICE, O_RDWR);
    if (arena->fd < 0 || ioctl(arena->fd, BCUDA_XLNK_IOCALLOCBUF, request) != 0) goto fail;
    memcpy(&arena->buffer_id, request + 4, sizeof(arena->buffer_id));
    memcpy(&arena->physical_address, request + 8, sizeof(arena->physical_address));
    if (arena->buffer_id <= 0 || arena->physical_address == 0) goto fail;
    arena->bytes = requested;
    arena->virtual_address = mmap(0, requested, PROT_READ | PROT_WRITE, MAP_SHARED, arena->fd, (off_t)arena->buffer_id << 16);
    if (arena->virtual_address != MAP_FAILED) return 0;
fail:
    if (arena->fd >= 0) close(arena->fd);
    arena->fd = -1;
    return -1;
}

static void cma_free(BcudaCmaArena *arena) {
    if (arena->virtual_address != 0 && arena->virtual_address != MAP_FAILED) munmap(arena->virtual_address, arena->bytes);
    if (arena->fd >= 0) close(arena->fd);
}

int main(int argc, char **argv) {
    uint16_t port = BCUDA_PROTOCOL_DEFAULT_PORT;
    uint32_t arena_mib = BCUDA_DEFAULT_ARENA_MIB;
    int mem_fd;
    void *registers;
    BcudaCmaArena arena;
    BcudaAxiDma dma;
    BcudaGraphRuntime runtime;

    if (argc > 1) port = (uint16_t)strtoul(argv[1], 0, 0);
    if (argc > 2) arena_mib = (uint32_t)strtoul(argv[2], 0, 0);
    if (port == 0 || arena_mib == 0 || (uint64_t)arena_mib * 1024u * 1024u > UINT32_MAX) {
        fprintf(stderr, "usage: %s [tcp-port] [cma-arena-mib]\n", argv[0]);
        return 2;
    }
    if (cma_alloc(&arena, (size_t)arena_mib * 1024u * 1024u) != 0) {
        fprintf(stderr, "BudgetCUDA: cannot allocate non-cacheable CMA arena from %s: %s\n", BCUDA_XLNK_DEVICE, strerror(errno));
        return 1;
    }
    mem_fd = open(BCUDA_MEM_DEVICE, O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        fprintf(stderr, "BudgetCUDA: cannot open %s: %s\n", BCUDA_MEM_DEVICE, strerror(errno));
        cma_free(&arena);
        return 1;
    }
    registers = mmap(0, BCUDA_MMIO_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, BCUDA_PL_DEFAULT_REGISTER_BASE);
    close(mem_fd);
    if (registers == MAP_FAILED) {
        fprintf(stderr, "BudgetCUDA: cannot map PL AXI-Lite registers: %s\n", strerror(errno));
        cma_free(&arena);
        return 1;
    }
    bcuda_buffer_manager_init_with_physical(arena.virtual_address, arena.physical_address, arena.bytes);
    bcuda_axi_dma_init(&dma, (uintptr_t)registers, 10000000u);
    bcuda_graph_runtime_init(&runtime, &dma);
    fprintf(stderr, "BudgetCUDA: serving on TCP %u with %u MiB CMA at 0x%08x\n", port, arena_mib, arena.physical_address);
    bcuda_protocol_server_run(port, &runtime);
    munmap(registers, BCUDA_MMIO_BYTES);
    cma_free(&arena);
    return 1;
}
