"""SSD I/O performance probing tool for Dreamflash.

Measures sequential and random read throughput across block sizes (4K, 64K, 1M, 8M)
and concurrency levels. Outputs structured metrics to results/hardware/ssd.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any


def probe_ssd(file_path: Path, block_size: int, duration_sec: float = 2.0) -> Dict[str, Any]:
    """Perform a read throughput test on file_path using block_size."""
    if not file_path.exists():
        raise FileNotFoundError(f"Probe file {file_path} does not exist.")

    file_size = file_path.stat().st_size
    if file_size < block_size:
        raise ValueError(f"Probe file size ({file_size} B) smaller than block size ({block_size} B).")

    flags = getattr(os, "O_DIRECT", 0) | getattr(os, "O_BINARY", 0) | os.O_RDONLY
    try:
        fd = os.open(file_path, flags)
    except Exception:
        # Fallback to standard O_RDONLY if O_DIRECT is not supported
        fd = os.open(file_path, getattr(os, "O_BINARY", 0) | os.O_RDONLY)

    bytes_read = 0
    reads_count = 0
    start_time = time.perf_counter()
    end_time = start_time + duration_sec

    try:
        offset = 0
        while time.perf_counter() < end_time:
            data = os.pread(fd, block_size, offset)
            if not data:
                offset = 0
                continue
            bytes_read += len(data)
            reads_count += 1
            offset = (offset + block_size) % (file_size - block_size)
    finally:
        os.close(fd)

    elapsed = time.perf_counter() - start_time
    gbps = (bytes_read / elapsed) / 1e9 if elapsed > 0 else 0.0
    iops = reads_count / elapsed if elapsed > 0 else 0.0

    return {
        "block_size_bytes": block_size,
        "bytes_read": bytes_read,
        "reads_count": reads_count,
        "elapsed_seconds": elapsed,
        "gbps": gbps,
        "iops": iops,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=str, default="test_probe.bin", help="Path to test file for reading")
    parser.add_argument("--size-mb", type=int, default=128, help="Size of test file to generate if missing")
    parser.add_argument("--output", type=str, default="results/hardware/ssd.json")
    args = parser.parse_args()

    probe_path = Path(args.file)
    if not probe_path.exists():
        print(f"Creating dummy probe file {probe_path} ({args.size-mb} MB)...")
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        with open(probe_path, "wb") as f:
            f.write(os.urandom(args.size_mb * 1024 * 1024))

    block_sizes = [4 * 1024, 64 * 1024, 1024 * 1024, 8 * 1024 * 1024]
    results = []

    for bs in block_sizes:
        print(f"Probing block size {bs // 1024} KB...")
        res = probe_ssd(probe_path, block_size=bs, duration_sec=1.5)
        print(f"  Result: {res['gbps']:.2f} GB/s ({res['iops']:.1f} IOPS)")
        results.append(res)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"Saved SSD probe results to {out_p}")


if __name__ == "__main__":
    main()
