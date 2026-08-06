/**
 * Dreamflash Three-Tier Memory Manager Implementation.
 */

#include "memory_manager.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_EXPERTS_PER_LAYER 512
#define MAX_LAYERS 64
#define TOTAL_EXPERT_KEYS (MAX_LAYERS * MAX_EXPERTS_PER_LAYER)

typedef struct {
    int in_vram;
    int in_host;
    uint64_t last_used_step;
} expert_entry_t;

struct ds4_memory_manager {
    uint32_t vram_capacity;
    uint32_t host_capacity;
    uint32_t vram_count;
    uint32_t host_count;

    ds4_memory_stats_t stats;

    expert_entry_t entries[TOTAL_EXPERT_KEYS];
};

/* Returns the table slot for `key`, or KEY_INDEX_INVALID if the key is out of range.
 * Folding out-of-range keys onto slot 0 (as this previously did) makes every bad key
 * alias {layer 0, expert 0} and report a false cache hit -- with MAX_LAYERS=64 and
 * 512 experts/layer against a 43x256 model, that is a live hazard, not a theoretical
 * one. Callers must check. */
#define KEY_INDEX_INVALID ((size_t)-1)

static inline size_t key_index(expert_key_t key) {
    if (key.layer_idx >= MAX_LAYERS || key.expert_idx >= MAX_EXPERTS_PER_LAYER) {
        return KEY_INDEX_INVALID;
    }
    return (size_t)key.layer_idx * MAX_EXPERTS_PER_LAYER + key.expert_idx;
}

ds4_memory_manager_t *ds4_memory_manager_init(uint32_t vram_capacity, uint32_t host_capacity) {
    ds4_memory_manager_t *mgr = (ds4_memory_manager_t *)calloc(1, sizeof(ds4_memory_manager_t));
    if (!mgr) return NULL;

    mgr->vram_capacity = vram_capacity;
    mgr->host_capacity = host_capacity;
    return mgr;
}

ds4_cache_hit_tier_t ds4_memory_manager_lookup(ds4_memory_manager_t *mgr, expert_key_t key, uint64_t step) {
    if (!mgr) return EXPERT_CACHE_MISS_SSD;

    size_t idx = key_index(key);
    if (idx == KEY_INDEX_INVALID) return EXPERT_CACHE_MISS_SSD;
    expert_entry_t *entry = &mgr->entries[idx];

    if (entry->in_vram) {
        entry->last_used_step = step;
        return EXPERT_CACHE_HIT_VRAM;
    } else if (entry->in_host) {
        entry->last_used_step = step;
        return EXPERT_CACHE_HIT_HOST_RAM;
    }

    return EXPERT_CACHE_MISS_SSD;
}

static size_t find_lru_vram_index(ds4_memory_manager_t *mgr) {
    size_t min_idx = 0;
    uint64_t min_step = (uint64_t)-1;

    for (size_t i = 0; i < TOTAL_EXPERT_KEYS; i++) {
        if (mgr->entries[i].in_vram && mgr->entries[i].last_used_step < min_step) {
            min_step = mgr->entries[i].last_used_step;
            min_idx = i;
        }
    }
    return min_idx;
}

static size_t find_lru_host_index(ds4_memory_manager_t *mgr) {
    size_t min_idx = 0;
    uint64_t min_step = (uint64_t)-1;

    for (size_t i = 0; i < TOTAL_EXPERT_KEYS; i++) {
        if (mgr->entries[i].in_host && mgr->entries[i].last_used_step < min_step) {
            min_step = mgr->entries[i].last_used_step;
            min_idx = i;
        }
    }
    return min_idx;
}

int ds4_memory_manager_insert(ds4_memory_manager_t *mgr, expert_key_t key, uint64_t step) {
    if (!mgr) return -1;

    size_t idx = key_index(key);
    if (idx == KEY_INDEX_INVALID) return -1;
    expert_entry_t *entry = &mgr->entries[idx];

    /* No tier configured: nothing can be cached. Report it rather than returning
     * success having stored nothing, which callers cannot distinguish from a hit. */
    if (mgr->vram_capacity == 0 && mgr->host_capacity == 0) return -1;

    if (entry->in_vram) {
        entry->last_used_step = step;
        return 0;
    }

    if (entry->in_host) {
        entry->in_host = 0;
        if (mgr->host_count > 0) mgr->host_count--;
    }

    // Insert into VRAM
    if (mgr->vram_capacity > 0) {
        if (mgr->vram_count >= mgr->vram_capacity) {
            // Evict LRU from VRAM -> demote to Host RAM
            size_t lru_vram = find_lru_vram_index(mgr);
            mgr->entries[lru_vram].in_vram = 0;
            mgr->vram_count--;
            mgr->stats.vram_evictions++;

            if (mgr->host_capacity > 0) {
                if (mgr->host_count >= mgr->host_capacity) {
                    // Evict LRU from Host RAM -> falls out of the cache entirely
                    size_t lru_host = find_lru_host_index(mgr);
                    mgr->entries[lru_host].in_host = 0;
                    mgr->host_count--;
                    mgr->stats.host_evictions++;
                    mgr->stats.dropped_to_ssd++;
                }
                mgr->entries[lru_vram].in_host = 1;
                mgr->host_count++;
            } else {
                /* No host tier to demote into: this expert leaves the cache and will
                 * need an SSD refetch. Distinct from a demotion, and callers cannot
                 * see the difference from the return value alone. */
                mgr->stats.dropped_to_ssd++;
            }
        }

        entry->in_vram = 1;
        entry->last_used_step = step;
        mgr->vram_count++;
    } else if (mgr->host_capacity > 0) {
        if (mgr->host_count >= mgr->host_capacity) {
            size_t lru_host = find_lru_host_index(mgr);
            mgr->entries[lru_host].in_host = 0;
            mgr->host_count--;
            mgr->stats.host_evictions++;
            mgr->stats.dropped_to_ssd++;
        }
        entry->in_host = 1;
        entry->last_used_step = step;
        mgr->host_count++;
    }

    return 0;
}

int ds4_memory_manager_get_stats(const ds4_memory_manager_t *mgr, ds4_memory_stats_t *out) {
    if (!mgr || !out) return -1;
    *out = mgr->stats;
    out->vram_resident = mgr->vram_count;
    out->host_resident = mgr->host_count;
    return 0;
}

void ds4_memory_manager_free(ds4_memory_manager_t *mgr) {
    if (mgr) free(mgr);
}
