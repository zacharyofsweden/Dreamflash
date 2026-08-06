/**
 * Dreamflash Expert I/O Engine Implementation.
 *
 * Win32: genuine overlapped I/O (ReadFile + GetOverlappedResult).
 * Linux/WSL2: synchronous pread, looped so short reads are handled rather than
 * silently reported as complete. io_uring is NOT implemented; requesting it
 * downgrades to pread and expert_io_backend_in_use() reports the downgrade.
 */

/* O_DIRECT is a GNU extension: without this it is simply not declared, and the
 * #ifdef guard below silently compiles out, leaving every read going through the
 * page cache. Must precede any libc header. */
#if !defined(_WIN32) && !defined(_WIN64)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include "expert_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>

struct expert_io_context {
    HANDLE file_handle;
    expert_io_backend_type_t backend_type;
    uint32_t max_queue_depth;
    int unbuffered;
};

expert_io_context_t *expert_io_init(
    const char *file_path,
    expert_io_backend_type_t backend_type,
    uint32_t max_queue_depth
) {
    if (!file_path) return NULL;

    expert_io_context_t *ctx = (expert_io_context_t *)calloc(1, sizeof(expert_io_context_t));
    if (!ctx) return NULL;

    /* Win32 has exactly one backend here; record what the caller actually gets
     * rather than echoing back what they asked for. */
    (void)backend_type;
    ctx->backend_type = EXPERT_IO_BACKEND_WIN32;
    ctx->max_queue_depth = max_queue_depth;

    HANDLE hFile = CreateFileA(
        file_path,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
        NULL
    );
    ctx->unbuffered = 1;

    /* Only retry buffered if unbuffered mode itself was the problem. A missing file
     * or a permissions failure will fail identically the second time, and retrying
     * just hides which error actually occurred. */
    if (hFile == INVALID_HANDLE_VALUE) {
        DWORD err = GetLastError();
        if (err == ERROR_INVALID_PARAMETER || err == ERROR_NOT_SUPPORTED) {
            hFile = CreateFileA(
                file_path,
                GENERIC_READ,
                FILE_SHARE_READ,
                NULL,
                OPEN_EXISTING,
                FILE_FLAG_OVERLAPPED,
                NULL
            );
            ctx->unbuffered = 0;
        }
    }

    if (hFile == INVALID_HANDLE_VALUE) {
        free(ctx);
        return NULL;
    }

    ctx->file_handle = hFile;
    return ctx;
}

expert_io_backend_type_t expert_io_backend_in_use(const expert_io_context_t *ctx) {
    return ctx ? ctx->backend_type : EXPERT_IO_BACKEND_PREAD;
}

int expert_io_is_unbuffered(const expert_io_context_t *ctx) {
    return ctx ? ctx->unbuffered : 0;
}

void *expert_io_alloc_aligned(size_t length_bytes) {
    return _aligned_malloc(length_bytes, EXPERT_IO_ALIGNMENT);
}

void expert_io_free_aligned(void *buffer) {
    if (buffer) _aligned_free(buffer);
}

void expert_io_release(expert_io_request_t *req) {
    if (!req) return;
    if (req->internal) {
        free(req->internal);
        req->internal = NULL;
    }
}

int expert_io_submit(expert_io_context_t *ctx, expert_io_request_t *req) {
    if (!ctx || !req || ctx->file_handle == INVALID_HANDLE_VALUE) return -1;
    if (!req->destination_buffer || req->length_bytes == 0) {
        if (req) req->error_code = ERROR_INVALID_PARAMETER;
        return -1;
    }

    if (ctx->unbuffered) {
        uintptr_t addr = (uintptr_t)req->destination_buffer;
        if ((addr % EXPERT_IO_ALIGNMENT) != 0 ||
            (req->length_bytes % EXPERT_IO_ALIGNMENT) != 0 ||
            (req->file_offset % EXPERT_IO_ALIGNMENT) != 0) {
            req->error_code = ERROR_INVALID_PARAMETER;
            req->is_completed = 0;
            return -1;
        }
    }

    /* Reuse any OVERLAPPED already attached, so a resubmitted request does not leak. */
    OVERLAPPED *ov = (OVERLAPPED *)req->internal;
    if (!ov) {
        ov = (OVERLAPPED *)calloc(1, sizeof(OVERLAPPED));
        if (!ov) {
            req->error_code = ERROR_NOT_ENOUGH_MEMORY;
            return -1;
        }
        req->internal = ov;
    } else {
        memset(ov, 0, sizeof(OVERLAPPED));
    }

    ULARGE_INTEGER uli;
    uli.QuadPart = req->file_offset;
    ov->Offset = uli.LowPart;
    ov->OffsetHigh = uli.HighPart;

    req->is_completed = 0;
    req->error_code = 0;
    req->bytes_transferred = 0;

    DWORD immediate = 0;
    BOOL ok = ReadFile(
        ctx->file_handle,
        req->destination_buffer,
        (DWORD)req->length_bytes,
        &immediate,
        ov
    );

    if (ok) {
        /* Completed synchronously. */
        req->bytes_transferred = (size_t)immediate;
        req->is_completed = 1;
        return 0;
    }

    if (GetLastError() == ERROR_IO_PENDING) {
        /* Still in flight. The buffer must not be touched until expert_io_wait(). */
        return 0;
    }

    req->error_code = (int)GetLastError();
    expert_io_release(req);
    return -1;
}

int expert_io_wait(expert_io_context_t *ctx, expert_io_request_t *req, uint32_t timeout_ms) {
    if (!ctx || !req) return -1;
    if (req->is_completed) return 0;

    OVERLAPPED *ov = (OVERLAPPED *)req->internal;
    if (!ov) return -1;

    /* Bounded wait on the event-less OVERLAPPED: poll for completion, then collect
     * the real transfer count. Returning "completed" without this is a data race --
     * the kernel may still be writing into destination_buffer. */
    DWORD wait_rc = WaitForSingleObject(ctx->file_handle, timeout_ms);
    if (wait_rc == WAIT_TIMEOUT) {
        return 1;
    }

    DWORD transferred = 0;
    BOOL ok = GetOverlappedResult(ctx->file_handle, ov, &transferred, TRUE);
    if (!ok) {
        DWORD err = GetLastError();
        if (err == ERROR_HANDLE_EOF) {
            req->bytes_transferred = (size_t)transferred;
            req->is_completed = 1;
            req->error_code = 0;
            expert_io_release(req);
            return 0;
        }
        req->error_code = (int)err;
        expert_io_release(req);
        return -1;
    }

    req->bytes_transferred = (size_t)transferred;
    req->is_completed = 1;
    req->error_code = 0;
    expert_io_release(req);
    return 0;
}

void expert_io_close(expert_io_context_t *ctx) {
    if (!ctx) return;
    if (ctx->file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(ctx->file_handle);
        ctx->file_handle = INVALID_HANDLE_VALUE;
    }
    free(ctx);
}

#else
/* Linux / POSIX Implementation */
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

struct expert_io_context {
    int fd;
    expert_io_backend_type_t backend_type;
    uint32_t max_queue_depth;
    int unbuffered;
};

expert_io_context_t *expert_io_init(
    const char *file_path,
    expert_io_backend_type_t backend_type,
    uint32_t max_queue_depth
) {
    if (!file_path) return NULL;

    expert_io_context_t *ctx = (expert_io_context_t *)calloc(1, sizeof(expert_io_context_t));
    if (!ctx) return NULL;

    int fd = -1;
    int unbuffered = 0;

#ifdef O_DIRECT
    fd = open(file_path, O_RDONLY | O_DIRECT);
    if (fd >= 0) {
        unbuffered = 1;
    } else if (errno == EINVAL || errno == ENOTSUP || errno == EOPNOTSUPP) {
        /* Filesystem does not support O_DIRECT (tmpfs, some overlayfs). Retry
         * buffered. Any other errno -- ENOENT, EACCES -- will fail identically
         * buffered, so let it through rather than masking the real cause. */
        fd = open(file_path, O_RDONLY);
    }
#else
    fd = open(file_path, O_RDONLY);
#endif

    if (fd < 0) {
        free(ctx);
        return NULL;
    }

    ctx->fd = fd;
    ctx->unbuffered = unbuffered;
    ctx->max_queue_depth = max_queue_depth;
    /* io_uring is not implemented. Report the backend actually in use so callers
     * are not misled into assuming submissions are asynchronous. */
    ctx->backend_type = (backend_type == EXPERT_IO_BACKEND_IO_URING)
                            ? EXPERT_IO_BACKEND_PREAD
                            : backend_type;
    return ctx;
}

expert_io_backend_type_t expert_io_backend_in_use(const expert_io_context_t *ctx) {
    return ctx ? ctx->backend_type : EXPERT_IO_BACKEND_PREAD;
}

int expert_io_is_unbuffered(const expert_io_context_t *ctx) {
    return ctx ? ctx->unbuffered : 0;
}

void *expert_io_alloc_aligned(size_t length_bytes) {
    void *p = NULL;
    /* posix_memalign wants a length that is a multiple of the alignment for the
     * buffer to be usable with O_DIRECT end-to-end. */
    size_t rounded = (length_bytes + EXPERT_IO_ALIGNMENT - 1) & ~((size_t)EXPERT_IO_ALIGNMENT - 1);
    if (posix_memalign(&p, EXPERT_IO_ALIGNMENT, rounded) != 0) return NULL;
    return p;
}

void expert_io_free_aligned(void *buffer) {
    free(buffer);
}

void expert_io_release(expert_io_request_t *req) {
    if (!req) return;
    req->internal = NULL; /* pread backend attaches no private state */
}

int expert_io_submit(expert_io_context_t *ctx, expert_io_request_t *req) {
    if (!ctx || !req || ctx->fd < 0) return -1;
    if (!req->destination_buffer || req->length_bytes == 0) {
        req->error_code = EINVAL;
        return -1;
    }

    if (ctx->unbuffered) {
        uintptr_t addr = (uintptr_t)req->destination_buffer;
        if ((addr % EXPERT_IO_ALIGNMENT) != 0 ||
            (req->length_bytes % EXPERT_IO_ALIGNMENT) != 0 ||
            (req->file_offset % EXPERT_IO_ALIGNMENT) != 0) {
            /* O_DIRECT would reject this with EINVAL deep in the kernel; fail here
             * where the cause is obvious. */
            req->error_code = EINVAL;
            req->is_completed = 0;
            return -1;
        }
    }

    req->is_completed = 0;
    req->error_code = 0;
    req->bytes_transferred = 0;

    /* pread is permitted to return fewer bytes than requested even without EOF.
     * Loop until the request is satisfied, EOF, or a hard error. */
    size_t total = 0;
    while (total < req->length_bytes) {
        ssize_t ret = pread(
            ctx->fd,
            (char *)req->destination_buffer + total,
            req->length_bytes - total,
            (off_t)(req->file_offset + total)
        );

        if (ret > 0) {
            total += (size_t)ret;
            continue;
        }
        if (ret == 0) {
            break; /* EOF: short read, reported via bytes_transferred */
        }
        if (errno == EINTR) {
            continue;
        }

        req->error_code = errno; /* errno, not the -1 return value */
        req->bytes_transferred = total;
        return -1;
    }

    req->bytes_transferred = total;
    req->is_completed = 1;
    req->error_code = 0;
    return 0;
}

int expert_io_wait(expert_io_context_t *ctx, expert_io_request_t *req, uint32_t timeout_ms) {
    (void)ctx;
    (void)timeout_ms; /* pread completes during submit; nothing to wait for */
    if (!req) return -1;
    return req->is_completed ? 0 : -1;
}

void expert_io_close(expert_io_context_t *ctx) {
    if (!ctx) return;
    if (ctx->fd >= 0) {
        close(ctx->fd);
        ctx->fd = -1;
    }
    free(ctx);
}

#endif
