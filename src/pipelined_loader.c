/**
 * Dreamflash Pipelined Double-Buffered Expert Loader Implementation.
 */

#include "pipelined_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct pipelined_loader {
    expert_io_context_t *io_ctx;
    ds4_memory_manager_t *mem_mgr;

    size_t slot_capacity;
    void *buffer_A;
    void *buffer_B;
    int active_buffer_idx; // 0 for A, 1 for B

    expert_io_request_t req_inflight;
    expert_key_t current_key;
    int has_inflight;
};

pipelined_loader_t *pipelined_loader_init(
    expert_io_context_t *io_ctx,
    ds4_memory_manager_t *mem_mgr,
    size_t slot_capacity_bytes
) {
    if (!io_ctx || !mem_mgr || slot_capacity_bytes == 0) return NULL;

    pipelined_loader_t *loader = (pipelined_loader_t *)calloc(1, sizeof(pipelined_loader_t));
    if (!loader) return NULL;

    loader->io_ctx = io_ctx;
    loader->mem_mgr = mem_mgr;
    loader->slot_capacity = slot_capacity_bytes;

    loader->buffer_A = expert_io_alloc_aligned(slot_capacity_bytes);
    loader->buffer_B = expert_io_alloc_aligned(slot_capacity_bytes);

    if (!loader->buffer_A || !loader->buffer_B) {
        pipelined_loader_free(loader);
        return NULL;
    }

    loader->active_buffer_idx = 0;
    loader->has_inflight = 0;
    return loader;
}

int pipelined_loader_prefetch_next(
    pipelined_loader_t *loader,
    expert_key_t next_key,
    uint64_t file_offset,
    size_t length_bytes
) {
    if (!loader) return -1;
    if (length_bytes > loader->slot_capacity) return -1;

    // Check if expert is already in VRAM or RAM cache
    ds4_cache_hit_tier_t tier = ds4_memory_manager_lookup(loader->mem_mgr, next_key, 0);
    if (tier == EXPERT_CACHE_HIT_VRAM) {
        // VRAM hit: zero SSD/PCIe read required
        return 0;
    }

    void *target_staging = (loader->active_buffer_idx == 0) ? loader->buffer_A : loader->buffer_B;
    loader->active_buffer_idx = 1 - loader->active_buffer_idx;

    memset(&loader->req_inflight, 0, sizeof(expert_io_request_t));
    loader->req_inflight.layer_idx = next_key.layer_idx;
    loader->req_inflight.expert_idx = next_key.expert_idx;
    loader->req_inflight.file_offset = file_offset;
    loader->req_inflight.length_bytes = length_bytes;
    loader->req_inflight.destination_buffer = target_staging;

    loader->current_key = next_key;
    loader->has_inflight = 1;

    return expert_io_submit(loader->io_ctx, &loader->req_inflight);
}

int pipelined_loader_commit_current(pipelined_loader_t *loader, uint64_t step) {
    if (!loader) return -1;
    if (!loader->has_inflight) return 0;

    int rc = expert_io_wait(loader->io_ctx, &loader->req_inflight, 1000);
    if (rc == 0) {
        ds4_memory_manager_insert(loader->mem_mgr, loader->current_key, step);
        loader->has_inflight = 0;
        return 0;
    }
    return -1;
}

void pipelined_loader_free(pipelined_loader_t *loader) {
    if (!loader) return;
    if (loader->buffer_A) expert_io_free_aligned(loader->buffer_A);
    if (loader->buffer_B) expert_io_free_aligned(loader->buffer_B);
    free(loader);
}
