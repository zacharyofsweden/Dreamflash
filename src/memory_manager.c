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

    expert_entry_t entries[TOTAL_EXPERT_KEYS];
};

static inline size_t key_index(expert_key_t key) {
    if (key.layer_idx >= MAX_LAYERS || key.expert_idx >= MAX_EXPERTS_PER_LAYER) {
        return 0;
    }
    return key.layer_idx * MAX_EXPERTS_PER_LAYER + key.expert_idx;
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
    expert_entry_t *entry = &mgr->entries[idx];

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

            if (mgr->host_capacity > 0) {
                if (mgr->host_count >= mgr->host_capacity) {
                    // Evict LRU from Host RAM
                    size_t lru_host = find_lru_host_index(mgr);
                    mgr->entries[lru_host].in_host = 0;
                    mgr->host_count--;
                }
                mgr->entries[lru_vram].in_host = 1;
                mgr->host_count++;
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
        }
        entry->in_host = 1;
        entry->last_used_step = step;
        mgr->host_count++;
    }

    return 0;
}

void ds4_memory_manager_free(ds4_memory_manager_t *mgr) {
    if (mgr) free(mgr);
}
