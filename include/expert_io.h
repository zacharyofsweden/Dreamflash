/**
 * Dreamflash Expert IO Backend Interface.
 *
 * Backend interface for streaming expert parameters off disk:
 * - pread backend (POSIX)      -- SYNCHRONOUS: submit() performs the read and returns
 *                                 completed. wait() is then a no-op that reports status.
 * - Win32 ReadFile backend     -- genuinely asynchronous (FILE_FLAG_OVERLAPPED).
 * - io_uring backend           -- NOT IMPLEMENTED. Requesting it currently falls back
 *                                 to pread; expert_io_backend_in_use() reports what
 *                                 you actually got.
 *
 * Callers must not assume submit() is non-blocking. Check expert_io_backend_in_use()
 * if the distinction matters for your scheduling.
 *
 * O_DIRECT / FILE_FLAG_NO_BUFFERING: when unbuffered I/O is active, the destination
 * buffer address, length_bytes, and file_offset must all be multiples of
 * EXPERT_IO_ALIGNMENT. Use expert_io_alloc_aligned() for the buffer. Requests that
 * violate this are rejected with EINVAL rather than being passed to the kernel.
 */

#ifndef EXPERT_IO_H
#define EXPERT_IO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Alignment required for unbuffered I/O. Covers both 512e and 4Kn devices. */
#define EXPERT_IO_ALIGNMENT 4096u

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
    /** errno (POSIX) or GetLastError() (Win32) on failure; 0 on success. */
    int error_code;
    /**
     * Bytes actually delivered into destination_buffer. A completed request with
     * bytes_transferred < length_bytes is a SHORT READ (typically EOF) and the tail
     * of the buffer is untouched -- always check this, do not infer completion from
     * is_completed alone.
     */
    size_t bytes_transferred;
    /** Backend-private state. Do not touch; released by expert_io_wait/close. */
    void *internal;
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
 * Report which backend the context actually uses, which may differ from the one
 * requested (io_uring is not implemented and downgrades to pread).
 */
expert_io_backend_type_t expert_io_backend_in_use(const expert_io_context_t *ctx);

/** Report whether unbuffered (O_DIRECT / FILE_FLAG_NO_BUFFERING) I/O is active. */
int expert_io_is_unbuffered(const expert_io_context_t *ctx);

/**
 * Allocate a buffer suitable for unbuffered I/O. Free with expert_io_free_aligned().
 *
 * @return Aligned pointer, or NULL on failure
 */
void *expert_io_alloc_aligned(size_t length_bytes);

/** Release a buffer from expert_io_alloc_aligned(). NULL is accepted. */
void expert_io_free_aligned(void *buffer);

/**
 * Submit an expert read request.
 *
 * On the pread backend this performs the read synchronously and the request is
 * complete on return. On Win32 it may return with the read still in flight; call
 * expert_io_wait() before touching destination_buffer.
 *
 * On success, inspect req->bytes_transferred for short reads.
 *
 * @param ctx Valid I/O context
 * @param req Request structure describing expert file location & buffer
 * @return 0 on successful submission, -1 on error (req->error_code holds errno /
 *         GetLastError())
 */
int expert_io_submit(expert_io_context_t *ctx, expert_io_request_t *req);

/**
 * Wait for a submitted request to complete.
 *
 * Honours timeout_ms on the Win32 backend (0 = non-blocking poll). The pread backend
 * completes during submit, so this only reports status there.
 *
 * @param ctx Valid I/O context
 * @param req Request pointer to wait on
 * @return 0 on completed read, 1 on timeout with the request still pending,
 *         -1 on error (req->error_code is set)
 */
int expert_io_wait(expert_io_context_t *ctx, expert_io_request_t *req, uint32_t timeout_ms);

/**
 * Release backend-private state attached to a request that will never be waited on
 * (e.g. an abandoned submission). Safe to call more than once. Not needed after a
 * successful expert_io_wait(), which releases it.
 */
void expert_io_release(expert_io_request_t *req);

/**
 * Close I/O context and release open handles.
 *
 * All submitted requests must have been waited on or released first; outstanding
 * async reads write into caller buffers and are not tracked by the context.
 */
void expert_io_close(expert_io_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* EXPERT_IO_H */
