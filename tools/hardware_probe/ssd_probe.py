"""SSD read-throughput probe for Dreamflash.

Measures sequential and random read throughput across block sizes (4K, 64K, 1M, 8M),
cold and warm page cache, and writes structured metrics to results/hardware/ssd.json.

Page cache is the thing that makes naive versions of this tool lie. A 128 MB probe
file read in a loop is served entirely from RAM after the first pass, and reports
memcpy bandwidth as if it were NVMe bandwidth. Two defences here:

  * the probe file defaults to 4 GB, comfortably past any plausible cache residency
    for the target box, and
  * before each cold run we call posix_fadvise(POSIX_FADV_DONTNEED) over the whole
    file, which evicts its pages without needing root.

Cold and warm are both reported. If they are within a few percent of each other on
Linux, the eviction did not work and the numbers are not trustworthy -- the tool
says so rather than leaving you to notice.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# O_DIRECT requires the destination buffer, the length, and the file offset to all
# be aligned to the logical block size. 4096 covers both 512e and 4Kn devices.
ALIGNMENT = 4096


def _aligned_buffer(size: int, alignment: int = ALIGNMENT) -> memoryview:
    """Allocate a buffer whose base address is `alignment`-aligned.

    os.pread() returns a fresh unaligned heap bytes object, which O_DIRECT rejects
    with EINVAL. os.preadv() into an aligned mutable buffer is the way through.
    """
    raw = ctypes.create_string_buffer(size + alignment)
    offset = (-ctypes.addressof(raw)) % alignment
    return memoryview(raw)[offset : offset + size]


def _drop_file_cache(fd: int, file_size: int) -> bool:
    """Evict this file's pages from the page cache. Returns True if it likely worked."""
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None:
        return False
    try:
        os.fsync(fd)  # dirty pages are not droppable; flush first
        fadvise(fd, 0, file_size, dontneed)
        return True
    except OSError:
        return False


def probe_ssd(
    file_path: Path,
    block_size: int,
    duration_sec: float = 2.0,
    pattern: str = "sequential",
    use_direct: bool = True,
    cold: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Read from file_path in block_size units and report throughput and latency."""
    if not file_path.exists():
        raise FileNotFoundError(f"Probe file {file_path} does not exist.")

    file_size = file_path.stat().st_size
    if file_size < block_size * 2:
        raise ValueError(
            f"Probe file size ({file_size} B) too small for block size ({block_size} B)."
        )

    direct_flag = getattr(os, "O_DIRECT", 0)
    want_direct = bool(use_direct and direct_flag)
    flags = getattr(os, "O_BINARY", 0) | os.O_RDONLY | (direct_flag if want_direct else 0)

    direct_used = want_direct
    try:
        fd = os.open(file_path, flags)
    except OSError:
        fd = os.open(file_path, getattr(os, "O_BINARY", 0) | os.O_RDONLY)
        direct_used = False

    cache_dropped = _drop_file_cache(fd, file_size) if cold else False

    buf = _aligned_buffer(block_size)
    n_blocks = file_size // block_size
    rng = random.Random(seed)

    bytes_read = 0
    reads_count = 0
    latencies: List[float] = []
    short_reads = 0

    try:
        block_idx = 0
        start_time = time.perf_counter()
        end_time = start_time + duration_sec

        while time.perf_counter() < end_time:
            if pattern == "random":
                offset = rng.randrange(n_blocks) * block_size
            else:
                offset = (block_idx % n_blocks) * block_size
                block_idx += 1

            t0 = time.perf_counter()
            try:
                if hasattr(os, "preadv"):
                    got = os.preadv(fd, [buf], offset)
                elif hasattr(os, "pread"):
                    data = os.pread(fd, block_size, offset)
                    got = len(data)
                else:
                    os.lseek(fd, offset, os.SEEK_SET)
                    data = os.read(fd, block_size)
                    got = len(data)
            except OSError as e:
                if direct_used:
                    # Fall back to buffered rather than dying mid-run, and say so.
                    os.close(fd)
                    fd = os.open(file_path, getattr(os, "O_BINARY", 0) | os.O_RDONLY)
                    direct_used = False
                    continue
                raise RuntimeError(f"pread failed at offset {offset}: {e}") from e
            t1 = time.perf_counter()

            if got == 0:
                block_idx = 0
                continue
            if got < block_size:
                short_reads += 1

            latencies.append(t1 - t0)
            bytes_read += got
            reads_count += 1
    finally:
        os.close(fd)

    elapsed = time.perf_counter() - start_time
    gbps = (bytes_read / elapsed) / 1e9 if elapsed > 0 else 0.0
    iops = reads_count / elapsed if elapsed > 0 else 0.0

    latencies.sort()

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int(len(latencies) * p))
        return latencies[idx] * 1e6  # microseconds

    return {
        "block_size_bytes": block_size,
        "pattern": pattern,
        "cache": "cold" if cold else "warm",
        "o_direct": direct_used,
        "cache_dropped": cache_dropped,
        "file_size_bytes": file_size,
        "bytes_read": bytes_read,
        "reads_count": reads_count,
        "short_reads": short_reads,
        "elapsed_seconds": elapsed,
        "gbps": gbps,
        "iops": iops,
        "lat_p50_us": pct(0.50),
        "lat_p95_us": pct(0.95),
        "lat_p99_us": pct(0.99),
    }


def create_probe_file(path: Path, size_mb: int) -> None:
    """Write a probe file of the requested size, in chunks, without buffering it all."""
    print(f"Creating probe file {path} ({size_mb} MB)... this is a one-off cost.")
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk = os.urandom(8 * 1024 * 1024)
    written = 0
    total = size_mb * 1024 * 1024
    with open(path, "wb") as f:
        while written < total:
            n = min(len(chunk), total - written)
            f.write(chunk[:n])
            written += n
        f.flush()
        os.fsync(f.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=str, default="test_probe.bin", help="Probe file path")
    parser.add_argument(
        "--size-mb",
        type=int,
        default=4096,
        help="Size of probe file to generate if missing (default 4096 = 4 GB, "
        "sized to exceed page cache)",
    )
    parser.add_argument("--duration", type=float, default=1.5, help="Seconds per measurement")
    parser.add_argument("--no-direct", action="store_true", help="Skip O_DIRECT, use buffered I/O")
    parser.add_argument("--output", type=str, default="results/hardware/ssd.json")
    args = parser.parse_args()

    probe_path = Path(args.file)
    if not probe_path.exists():
        create_probe_file(probe_path, args.size_mb)

    file_gb = probe_path.stat().st_size / (1024**3)
    if file_gb < 2.0:
        print(
            f"[!] WARNING: probe file is only {file_gb:.2f} GB. If that fits in page cache, "
            f"the 'cold' numbers below are RAM bandwidth, not SSD bandwidth."
        )

    block_sizes = [4 * 1024, 64 * 1024, 1024 * 1024, 8 * 1024 * 1024]
    results = []

    for pattern in ("sequential", "random"):
        for bs in block_sizes:
            for cold in (True, False):
                label = f"{pattern:<10} {bs // 1024:>5} KB  {'cold' if cold else 'warm'}"
                print(f"Probing {label}...")
                res = probe_ssd(
                    probe_path,
                    block_size=bs,
                    duration_sec=args.duration,
                    pattern=pattern,
                    use_direct=not args.no_direct,
                    cold=cold,
                )
                print(
                    f"  {res['gbps']:6.2f} GB/s  {res['iops']:9.1f} IOPS  "
                    f"p50 {res['lat_p50_us']:7.1f} us  p99 {res['lat_p99_us']:8.1f} us  "
                    f"{'O_DIRECT' if res['o_direct'] else 'buffered'}"
                )
                results.append(res)

    # Sanity gate: if cold and warm agree closely, cache eviction did not happen.
    suspicious = []
    for cold_r in (r for r in results if r["cache"] == "cold"):
        warm_r = next(
            (
                w
                for w in results
                if w["cache"] == "warm"
                and w["block_size_bytes"] == cold_r["block_size_bytes"]
                and w["pattern"] == cold_r["pattern"]
            ),
            None,
        )
        if warm_r and warm_r["gbps"] > 0:
            ratio = cold_r["gbps"] / warm_r["gbps"]
            if ratio > 0.9 and not cold_r["o_direct"]:
                suspicious.append((cold_r["pattern"], cold_r["block_size_bytes"], ratio))

    trustworthy = not suspicious
    if suspicious:
        print(
            "\n[!] WARNING: cold and warm throughput are within 10% for "
            f"{len(suspicious)} configuration(s) with buffered I/O."
        )
        print("    Page cache was probably not evicted. Do NOT feed these numbers to")
        print("    roofline.py --measured. Try a larger --size-mb, or run with O_DIRECT.")

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(
            {
                "probe_file_bytes": probe_path.stat().st_size,
                "trustworthy": trustworthy,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nSaved SSD probe results to {out_p}")
    return 0 if trustworthy else 1


if __name__ == "__main__":
    sys.exit(main())
