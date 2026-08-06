/**
 * Behavioural tests for the expert I/O engine and the three-tier memory manager.
 *
 * Build and run:
 *   make test-c
 * or:
 *   cc -std=c11 -Iinclude tests/test_expert_io.c src/expert_io.c src/memory_manager.c \
 *      -o build/test_expert_io && ./build/test_expert_io
 */

#include "expert_io.h"
#include "memory_manager.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        g_checks++;                                                          \
        if (!(cond)) {                                                       \
            printf("  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);       \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

/* ------------------------------------------------------------------ expert_io */

/* Relative to the current directory, so this works from any build tree. */
#define TEST_FILE "expert_io_testdata.bin"
#define TEST_FILE_BYTES (64u * 1024u)

static int write_test_file(void) {
    FILE *f = fopen(TEST_FILE, "wb");
    if (!f) return -1;
    for (unsigned i = 0; i < TEST_FILE_BYTES; i++) {
        unsigned char b = (unsigned char)(i & 0xFF);
        if (fwrite(&b, 1, 1, f) != 1) { fclose(f); return -1; }
    }
    fclose(f);
    return 0;
}

static void test_read_returns_correct_bytes(void) {
    printf("test_read_returns_correct_bytes\n");
    expert_io_context_t *ctx = expert_io_init(TEST_FILE, EXPERT_IO_BACKEND_PREAD, 8);
    CHECK(ctx != NULL, "context should open");
    if (!ctx) return;

    size_t len = EXPERT_IO_ALIGNMENT;
    void *buf = expert_io_alloc_aligned(len);
    CHECK(buf != NULL, "aligned alloc should succeed");
    CHECK(((uintptr_t)buf % EXPERT_IO_ALIGNMENT) == 0, "buffer must be aligned");

    expert_io_request_t req;
    memset(&req, 0, sizeof(req));
    req.file_offset = EXPERT_IO_ALIGNMENT;
    req.length_bytes = len;
    req.destination_buffer = buf;

    int rc = expert_io_submit(ctx, &req);
    CHECK(rc == 0, "submit should succeed");
    CHECK(expert_io_wait(ctx, &req, 1000) == 0, "wait should report completion");
    CHECK(req.bytes_transferred == len, "should transfer the full request");
    CHECK(req.error_code == 0, "error_code should be 0 on success");

    /* File byte at offset i is (i & 0xFF), so the read window must match. */
    const unsigned char *got = (const unsigned char *)buf;
    int content_ok = 1;
    for (size_t i = 0; i < len; i++) {
        if (got[i] != (unsigned char)((EXPERT_IO_ALIGNMENT + i) & 0xFF)) { content_ok = 0; break; }
    }
    CHECK(content_ok, "buffer contents should match file contents at that offset");

    expert_io_free_aligned(buf);
    expert_io_close(ctx);
}

static void test_short_read_is_visible(void) {
    printf("test_short_read_is_visible\n");
    /* Buffered mode so we can straddle EOF without alignment constraints. */
    expert_io_context_t *ctx = expert_io_init(TEST_FILE, EXPERT_IO_BACKEND_PREAD, 8);
    CHECK(ctx != NULL, "context should open");
    if (!ctx) return;

    size_t len = 8192;
    void *buf = expert_io_alloc_aligned(len);
    memset(buf, 0xAA, len);

    /* Start 4096 bytes before EOF and ask for 8192: only half can be delivered. */
    expert_io_request_t req;
    memset(&req, 0, sizeof(req));
    req.file_offset = TEST_FILE_BYTES - 4096;
    req.length_bytes = len;
    req.destination_buffer = buf;

    int rc = expert_io_submit(ctx, &req);
    CHECK(rc == 0, "submit past EOF should still succeed");
    if (!req.is_completed) {
        expert_io_wait(ctx, &req, 1000);
    }
    CHECK(req.is_completed == 1, "request should be marked complete");
    /* The regression this guards: a short read used to be indistinguishable from a
     * full one, because the API had no way to report the transfer count at all. */
    CHECK(req.bytes_transferred == 4096, "short read must report the real byte count");
    CHECK(req.bytes_transferred < req.length_bytes, "caller must be able to detect the shortfall");

    expert_io_free_aligned(buf);
    expert_io_close(ctx);
}

static void test_missing_file_fails(void) {
    printf("test_missing_file_fails\n");
    expert_io_context_t *ctx = expert_io_init(
        "definitely_does_not_exist.bin", EXPERT_IO_BACKEND_PREAD, 8);
    CHECK(ctx == NULL, "opening a missing file must fail, not fall back silently");
}

static void test_io_uring_downgrade_is_reported(void) {
    printf("test_io_uring_downgrade_is_reported\n");
    expert_io_context_t *ctx = expert_io_init(TEST_FILE, EXPERT_IO_BACKEND_IO_URING, 8);
    CHECK(ctx != NULL, "context should open");
    if (!ctx) return;
    /* io_uring is not implemented; the context must admit which backend it is. */
    CHECK(expert_io_backend_in_use(ctx) != EXPERT_IO_BACKEND_IO_URING,
          "must not claim an io_uring backend that does not exist");
    expert_io_close(ctx);
}

static void test_unaligned_request_rejected_when_unbuffered(void) {
    printf("test_unaligned_request_rejected_when_unbuffered\n");
    expert_io_context_t *ctx = expert_io_init(TEST_FILE, EXPERT_IO_BACKEND_PREAD, 8);
    CHECK(ctx != NULL, "context should open");
    if (!ctx) return;

    if (!expert_io_is_unbuffered(ctx)) {
        printf("  SKIP: O_DIRECT unavailable on this filesystem\n");
        expert_io_close(ctx);
        return;
    }

    void *buf = expert_io_alloc_aligned(EXPERT_IO_ALIGNMENT * 2);
    expert_io_request_t req;
    memset(&req, 0, sizeof(req));
    req.file_offset = 1; /* deliberately misaligned */
    req.length_bytes = EXPERT_IO_ALIGNMENT;
    req.destination_buffer = buf;

    CHECK(expert_io_submit(ctx, &req) == -1, "misaligned O_DIRECT request must be rejected");
    CHECK(req.error_code != 0, "rejection must set a real error code");

    expert_io_free_aligned(buf);
    expert_io_close(ctx);
}

/* ------------------------------------------------------------- memory_manager */

static void test_out_of_range_key_is_a_miss(void) {
    printf("test_out_of_range_key_is_a_miss\n");
    ds4_memory_manager_t *mgr = ds4_memory_manager_init(4, 4);
    CHECK(mgr != NULL, "manager should init");
    if (!mgr) return;

    expert_key_t in_range = {0, 0};
    CHECK(ds4_memory_manager_insert(mgr, in_range, 1) == 0, "in-range insert should succeed");
    CHECK(ds4_memory_manager_lookup(mgr, in_range, 2) == EXPERT_CACHE_HIT_VRAM,
          "inserted key should hit");

    /* The regression this guards: out-of-range keys used to fold onto slot 0 and
     * report a false VRAM hit for {layer 0, expert 0}. */
    expert_key_t out_of_range = {999, 7};
    CHECK(ds4_memory_manager_lookup(mgr, out_of_range, 3) == EXPERT_CACHE_MISS_SSD,
          "out-of-range key must miss, not alias onto slot 0");
    CHECK(ds4_memory_manager_insert(mgr, out_of_range, 4) == -1,
          "out-of-range insert must fail");

    ds4_memory_manager_free(mgr);
}

static void test_zero_capacity_insert_reports_failure(void) {
    printf("test_zero_capacity_insert_reports_failure\n");
    ds4_memory_manager_t *mgr = ds4_memory_manager_init(0, 0);
    CHECK(mgr != NULL, "manager should init");
    if (!mgr) return;

    expert_key_t k = {1, 1};
    /* Used to return 0 (success) having stored nothing. */
    CHECK(ds4_memory_manager_insert(mgr, k, 1) == -1,
          "insert with no capacity must report failure");
    CHECK(ds4_memory_manager_lookup(mgr, k, 2) == EXPERT_CACHE_MISS_SSD, "and must not be cached");

    ds4_memory_manager_free(mgr);
}

static void test_demotion_and_drop_are_distinguishable(void) {
    printf("test_demotion_and_drop_are_distinguishable\n");

    /* With a host tier, a VRAM eviction is a demotion: still cached, no SSD refetch. */
    ds4_memory_manager_t *tiered = ds4_memory_manager_init(1, 4);
    expert_key_t a = {0, 0};
    expert_key_t b = {0, 1};
    ds4_memory_manager_insert(tiered, a, 1);
    ds4_memory_manager_insert(tiered, b, 2); /* evicts `a` from VRAM -> host */
    CHECK(ds4_memory_manager_lookup(tiered, a, 3) == EXPERT_CACHE_HIT_HOST_RAM,
          "evicted-from-VRAM expert should be found in host RAM");

    ds4_memory_stats_t st;
    CHECK(ds4_memory_manager_get_stats(tiered, &st) == 0, "stats should be readable");
    CHECK(st.vram_evictions == 1, "one VRAM eviction expected");
    CHECK(st.dropped_to_ssd == 0, "a demotion is not a drop");
    ds4_memory_manager_free(tiered);

    /* Without a host tier, the same eviction drops the expert entirely. */
    ds4_memory_manager_t *flat = ds4_memory_manager_init(1, 0);
    ds4_memory_manager_insert(flat, a, 1);
    ds4_memory_manager_insert(flat, b, 2);
    CHECK(ds4_memory_manager_lookup(flat, a, 3) == EXPERT_CACHE_MISS_SSD,
          "with no host tier the victim leaves the cache");

    CHECK(ds4_memory_manager_get_stats(flat, &st) == 0, "stats should be readable");
    CHECK(st.vram_evictions == 1, "one VRAM eviction expected");
    /* The regression this guards: a dropped expert was previously indistinguishable
     * from a demoted one, so callers could not tell an SSD refetch was coming. */
    CHECK(st.dropped_to_ssd == 1, "a drop must be counted as such");
    ds4_memory_manager_free(flat);
}

static void test_lru_order_and_residency_counts(void) {
    printf("test_lru_order_and_residency_counts\n");
    ds4_memory_manager_t *mgr = ds4_memory_manager_init(2, 0);
    expert_key_t a = {0, 0}, b = {0, 1}, c = {0, 2};

    ds4_memory_manager_insert(mgr, a, 1);
    ds4_memory_manager_insert(mgr, b, 2);
    ds4_memory_manager_lookup(mgr, a, 3); /* `a` is now more recent than `b` */
    ds4_memory_manager_insert(mgr, c, 4); /* should evict `b`, the true LRU */

    CHECK(ds4_memory_manager_lookup(mgr, b, 5) == EXPERT_CACHE_MISS_SSD, "LRU victim is `b`");
    CHECK(ds4_memory_manager_lookup(mgr, a, 6) == EXPERT_CACHE_HIT_VRAM, "`a` was refreshed");
    CHECK(ds4_memory_manager_lookup(mgr, c, 7) == EXPERT_CACHE_HIT_VRAM, "`c` is resident");

    ds4_memory_stats_t st;
    ds4_memory_manager_get_stats(mgr, &st);
    CHECK(st.vram_resident == 2, "residency must not exceed capacity");
    CHECK(st.host_resident == 0, "no host tier configured");

    ds4_memory_manager_free(mgr);
}

int main(void) {
    if (write_test_file() != 0) {
        printf("could not create %s (cwd not writable?)\n", TEST_FILE);
        return 2;
    }

    test_read_returns_correct_bytes();
    test_short_read_is_visible();
    test_missing_file_fails();
    test_io_uring_downgrade_is_reported();
    test_unaligned_request_rejected_when_unbuffered();
    test_out_of_range_key_is_a_miss();
    test_zero_capacity_insert_reports_failure();
    test_demotion_and_drop_are_distinguishable();
    test_lru_order_and_residency_counts();

    remove(TEST_FILE);

    printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
