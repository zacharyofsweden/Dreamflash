/**
 * Dreamflash Discrete GPU Memory Configuration Header for DwarfStar (ds4).
 *
 * Overrides upstream unified-memory (Strix Halo APU) hardcoded reserves:
 * - Upstream rocm/ds4_rocm_runtime.cuh:1524 hardcodes 16 GiB free-reserve headroom.
 * - Upstream rocm/ds4_rocm_runtime.cuh:1525 hardcodes 8 GiB managed-KV floor.
 *
 * On discrete 12 GB VRAM GPUs, these defaults cause immediate OOM.
 * This header provides configurable overrides and discrete VRAM budgeting math.
 */

#ifndef DS4_DISCRETE_CONFIG_H
#define DS4_DISCRETE_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Default Discrete GPU Sizing Overrides (in bytes) */
#define DS4_DISCRETE_DEFAULT_VRAM_BYTES       (12ULL * 1024ULL * 1024ULL * 1024ULL)
#define DS4_DISCRETE_DEFAULT_RAM_BYTES        (16ULL * 1024ULL * 1024ULL * 1024ULL)
#define DS4_DISCRETE_FREE_RESERVE_VRAM_BYTES   (512ULL * 1024ULL * 1024ULL)       /* 512 MiB reserve */
#define DS4_DISCRETE_GRAPH_SCRATCH_BYTES       (1024ULL * 1024ULL * 1024ULL)      /* 1 GiB scratch */
#define DS4_DISCRETE_OS_RESERVE_RAM_BYTES      (3072ULL * 1024ULL * 1024ULL)      /* 3 GiB OS */
#define DS4_DISCRETE_STAGING_RAM_BYTES         (1536ULL * 1024ULL * 1024ULL)      /* 1.5 GiB staging */

typedef struct {
    uint64_t total_vram;
    uint64_t nonrouted_vram;
    uint64_t kv_vram;
    uint64_t scratch_vram;
    uint64_t free_reserve_vram;
    uint64_t available_vram_expert_cache;

    uint64_t total_ram;
    uint64_t os_reserve_ram;
    uint64_t staging_ram;
    uint64_t available_host_expert_cache;
} ds4_discrete_budget_t;

/**
 * Calculate discrete GPU VRAM and Host RAM budget for DeepSeek-V4 Flash.
 *
 * @param total_vram Total GPU VRAM in bytes (e.g. 12 GiB)
 * @param total_ram Total Host RAM in bytes (e.g. 16 GiB)
 * @param ctx_tokens Context length (e.g. 100,000)
 * @param kv_bytes_per_token KV state bytes per token (e.g. 20,000 B)
 * @param nonrouted_bytes Non-routed weights size (e.g. 6.9 GiB)
 * @param out_budget Pointer to output budget structure
 * @return 0 on success, -1 if mandatory state exceeds total VRAM
 */
static inline int ds4_calculate_discrete_budget(
    uint64_t total_vram,
    uint64_t total_ram,
    uint32_t ctx_tokens,
    uint64_t kv_bytes_per_token,
    uint64_t nonrouted_bytes,
    ds4_discrete_budget_t *out_budget
) {
    if (!out_budget) return -1;

    out_budget->total_vram = total_vram;
    out_budget->nonrouted_vram = nonrouted_bytes;
    out_budget->kv_vram = (uint64_t)ctx_tokens * kv_bytes_per_token;
    out_budget->scratch_vram = DS4_DISCRETE_GRAPH_SCRATCH_BYTES;
    out_budget->free_reserve_vram = DS4_DISCRETE_FREE_RESERVE_VRAM_BYTES;

    uint64_t mandatory_vram = out_budget->nonrouted_vram +
                              out_budget->kv_vram +
                              out_budget->scratch_vram +
                              out_budget->free_reserve_vram;

    if (mandatory_vram > total_vram) {
        out_budget->available_vram_expert_cache = 0;
    } else {
        out_budget->available_vram_expert_cache = total_vram - mandatory_vram;
    }

    out_budget->total_ram = total_ram;
    out_budget->os_reserve_ram = DS4_DISCRETE_OS_RESERVE_RAM_BYTES;
    out_budget->staging_ram = DS4_DISCRETE_STAGING_RAM_BYTES;

    uint64_t mandatory_ram = out_budget->os_reserve_ram + out_budget->staging_ram;

    if (mandatory_ram > total_ram) {
        out_budget->available_host_expert_cache = 0;
    } else {
        out_budget->available_host_expert_cache = total_ram - mandatory_ram;
    }

    return (mandatory_vram <= total_vram) ? 0 : -1;
}

#ifdef __cplusplus
}
#endif

#endif /* DS4_DISCRETE_CONFIG_H */
