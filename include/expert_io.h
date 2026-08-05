/**
 * Dreamflash Expert IO Backend Interface.
 *
 * Defines asynchronous NVMe I/O backend interface for streaming expert parameters from disk:
 * - io_uring backend (Linux kernel >= 5.1)
 * - pread fallback backend (POSIX)
 * - Win32 ReadFile backend (Windows)
 */

#ifndef EXPERT_IO_H
#define EXPERT_IO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    EXPERT_IO_BACKEND_PREAD = 0,
    EXPERT_IO_BACKEND_IO_URING = 1,
    EXPERT_IO_BACKEND_WIN32 = 2
} expert_io_backend_type_t;

typedef struct {
    uint32_t layer_idx;
    uint32_t expert_idx;
    uint64_t file_offset;
    size_t length_bytes;
    void *destination_buffer;
    volatile int is_completed;
    int error_code;
} expert_io_request_t;

typedef struct expert_io_context expert_io_context_t;

/**
 * Initialize an asynchronous expert I/O context.
 *
 * @param file_path Path to model weights file or GGUF container
 * @param backend_type Desired I/O backend mechanism
 * @param max_queue_depth Maximum inflight async requests
 * @return Context pointer, or NULL on failure
 */
expert_io_context_t *expert_io_init(
    const char *file_path,
    expert_io_backend_type_t backend_type,
    uint32_t max_queue_depth
);

/**
 * Submit an expert read request asynchronously.
 *
 * @param ctx Valid I/O context
 * @param req Request structure describing expert file location & buffer
 * @return 0 on successful submission, non-zero error code otherwise
 */
int expert_io_submit(expert_io_context_t *ctx, expert_io_request_t *req);

/**
 * Wait for a specific submitted request or drain queue up to timeout.
 *
 * @param ctx Valid I/O context
 * @param req Request pointer to wait on
 * @param timeout_ms Max wait time in milliseconds (0 for non-blocking poll)
 * @return 0 on completed read, -1 on timeout or error
 */
int expert_io_wait(expert_io_context_t *ctx, expert_io_request_t *req, uint32_t timeout_ms);

/**
 * Close I/O context and release open handles.
 */
void expert_io_close(expert_io_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* EXPERT_IO_H */
