/**
 * Dreamflash Three-Tier Expert Memory Manager Header.
 *
 * Manages expert allocation and eviction across 3 tiers:
 * Tier 1: VRAM Cache (fastest, GPU local)
 * Tier 2: Pinned Host RAM Cache (fast PCIe transfer)
 * Tier 3: NVMe SSD (disk storage)
 *
 * Scope: this tracks expert RESIDENCY as slot counts, not bytes. It holds no
 * buffers and performs no allocation of expert data; pair it with the byte budget
 * in ds4_discrete_config.h, which is what decides how many slots each tier gets.
 *
 * NOT THREAD SAFE. There is no internal locking, and lookup() mutates the entry it
 * finds (to record recency), so even the read path is a write. Callers sharing a
 * manager across threads -- e.g. alongside an async I/O engine -- must serialize
 * externally.
 */

#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    EXPERT_CACHE_HIT_VRAM = 0,
    EXPERT_CACHE_HIT_HOST_RAM = 1,
    EXPERT_CACHE_MISS_SSD = 2
} ds4_cache_hit_tier_t;

typedef struct {
    uint32_t layer_idx;
    uint32_t expert_idx;
} expert_key_t;

typedef struct {
    uint64_t vram_evictions;   /**< Experts evicted from VRAM (demoted or dropped). */
    uint64_t host_evictions;   /**< Experts evicted from host RAM. */
    uint64_t dropped_to_ssd;   /**< Experts that left the cache entirely and now need
                                *   an SSD refetch. Distinct from a VRAM->host demotion. */
    uint32_t vram_resident;    /**< Experts currently in VRAM. */
    uint32_t host_resident;    /**< Experts currently in host RAM. */
} ds4_memory_stats_t;

typedef struct ds4_memory_manager ds4_memory_manager_t;

/**
 * Initialize 3-tier memory manager.
 *
 * @param vram_capacity Number of experts VRAM tier can hold
 * @param host_capacity Number of experts Host RAM tier can hold
 * @return Context pointer, or NULL on failure
 */
ds4_memory_manager_t *ds4_memory_manager_init(uint32_t vram_capacity, uint32_t host_capacity);

/**
 * Lookup an expert key in cache tiers.
 *
 * An out-of-range key reports EXPERT_CACHE_MISS_SSD rather than aliasing onto
 * another slot. Mutates recency state -- see the thread-safety note above.
 *
 * @param mgr Memory manager instance
 * @param key Expert key (layer_idx, expert_idx)
 * @param step Access sequence step counter
 * @return EXPERT_CACHE_HIT_VRAM, EXPERT_CACHE_HIT_HOST_RAM, or EXPERT_CACHE_MISS_SSD
 */
ds4_cache_hit_tier_t ds4_memory_manager_lookup(ds4_memory_manager_t *mgr, expert_key_t key, uint64_t step);

/**
 * Insert or promote an expert into VRAM / Host RAM cache tiers.
 *
 * @param mgr Memory manager instance
 * @param key Expert key
 * @param step Access sequence step counter
 * @return 0 on success, -1 if the key is out of range or no tier has capacity (in
 *         which case nothing was stored)
 */
int ds4_memory_manager_insert(ds4_memory_manager_t *mgr, expert_key_t key, uint64_t step);

/**
 * Read residency and eviction counters.
 *
 * @return 0 on success, -1 on invalid arguments
 */
int ds4_memory_manager_get_stats(const ds4_memory_manager_t *mgr, ds4_memory_stats_t *out);

/**
 * Destroy memory manager and release internal tracking tables.
 */
void ds4_memory_manager_free(ds4_memory_manager_t *mgr);

#ifdef __cplusplus
}
#endif

#endif /* MEMORY_MANAGER_H */
