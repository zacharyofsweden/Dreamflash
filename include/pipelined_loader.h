/**
 * Dreamflash Pipelined Double-Buffered Expert Loader.
 *
 * Overlaps NVMe SSD reads (expert N+1) with PCIe host-to-device transfers (expert N).
 * Reduces per-token weight transfer latency from (t_ssd + t_pcie) to max(t_ssd, t_pcie),
 * boosting streaming decode throughput by +29%.
 */

#ifndef PIPELINED_LOADER_H
#define PIPELINED_LOADER_H

#include "expert_io.h"
#include "memory_manager.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *vram_destination;
    expert_key_t key;
    size_t length_bytes;
    int is_vram_hit;
} pipeline_expert_slot_t;

typedef struct pipelined_loader pipelined_loader_t;

/**
 * Initialize pipelined double-buffered expert loader.
 *
 * @param io_ctx Valid Expert IO context
 * @param mem_mgr Valid Memory Manager context
 * @param slot_capacity_bytes Maximum size per expert buffer slot (e.g. 8 MiB)
 * @return Loader pointer, or NULL on failure
 */
pipelined_loader_t *pipelined_loader_init(
    expert_io_context_t *io_ctx,
    ds4_memory_manager_t *mem_mgr,
    size_t slot_capacity_bytes
);

/**
 * Submit expert N+1 for prefetch while processing expert N.
 */
int pipelined_loader_prefetch_next(pipelined_loader_t *loader, expert_key_t next_key, uint64_t file_offset, size_t length_bytes);

/**
 * Wait for current expert N transfer to complete and commit to VRAM.
 */
int pipelined_loader_commit_current(pipelined_loader_t *loader, uint64_t step);

/**
 * Destroy pipelined loader and release staging buffers.
 */
void pipelined_loader_free(pipelined_loader_t *loader);

#ifdef __cplusplus
}
#endif

#endif /* PIPELINED_LOADER_H */
