"""PCIe host-to-device / device-to-host bandwidth probe.

Binds the HIP or CUDA runtime directly via ctypes and times real DMA transfers
between pinned host memory and device memory. There is no simulation path and no
fallback: if no GPU runtime is present, this tool reports failure and writes
nothing. A number that did not come off the bus is worse than no number, because
it will be quoted later as if it had.

  python tools/hardware_probe/pcie_probe.py            # pinned, both directions
  python tools/hardware_probe/pcie_probe.py --pageable # include pageable comparison

Reports GB/s per buffer size and direction, for pinned and (optionally) pageable
host memory. Decimal GB (1e9) throughout, to match roofline.py's bandwidth units.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Both runtimes expose the same enum values for these.
MEMCPY_HOST_TO_DEVICE = 1
MEMCPY_DEVICE_TO_HOST = 2

CANDIDATE_LIBS = [
    ("hip", "libamdhip64.so"),
    ("hip", "libamdhip64.so.6"),
    ("hip", "libamdhip64.so.5"),
    ("hip", "amdhip64.dll"),
    ("cuda", "libcudart.so"),
    ("cuda", "libcudart.so.12"),
    ("cuda", "libcudart.so.11.0"),
    ("cuda", "cudart64_12.dll"),
]


class GpuRuntime:
    """Thin ctypes binding over the handful of runtime calls this probe needs."""

    def __init__(self, kind: str, lib: ctypes.CDLL) -> None:
        self.kind = kind
        self.lib = lib
        p = "hip" if kind == "hip" else "cuda"
        self._malloc = getattr(lib, f"{p}Malloc")
        self._free = getattr(lib, f"{p}Free")
        self._host_alloc = getattr(lib, f"{p}HostMalloc" if kind == "hip" else "cudaMallocHost")
        self._host_free = getattr(lib, f"{p}HostFree" if kind == "hip" else "cudaFreeHost")
        self._memcpy = getattr(lib, f"{p}Memcpy")
        self._sync = getattr(lib, f"{p}DeviceSynchronize")
        self._get_count = getattr(lib, f"{p}GetDeviceCount")

        for fn in (self._malloc, self._host_alloc):
            fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            fn.restype = ctypes.c_int
        for fn in (self._free, self._host_free):
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_int
        self._memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self._memcpy.restype = ctypes.c_int
        self._sync.argtypes = []
        self._sync.restype = ctypes.c_int
        self._get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._get_count.restype = ctypes.c_int

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{self.kind} {what} failed with status {rc}")

    def device_count(self) -> int:
        n = ctypes.c_int(0)
        rc = self._get_count(ctypes.byref(n))
        if rc != 0:
            return 0
        return n.value

    def malloc_device(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self._check(self._malloc(ctypes.byref(ptr), nbytes), "device malloc")
        return ptr

    def malloc_host_pinned(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self._check(self._host_alloc(ctypes.byref(ptr), nbytes), "pinned host alloc")
        return ptr

    def free_device(self, ptr: ctypes.c_void_p) -> None:
        self._free(ptr)

    def free_host_pinned(self, ptr: ctypes.c_void_p) -> None:
        self._host_free(ptr)

    def memcpy(self, dst: ctypes.c_void_p, src: ctypes.c_void_p, nbytes: int, kind: int) -> None:
        self._check(self._memcpy(dst, src, nbytes, kind), "memcpy")

    def synchronize(self) -> None:
        self._check(self._sync(), "device synchronize")


def load_runtime() -> Optional[GpuRuntime]:
    """Locate and bind a HIP or CUDA runtime with at least one visible device."""
    for kind, name in CANDIDATE_LIBS:
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        try:
            rt = GpuRuntime(kind, lib)
        except AttributeError:
            continue
        if rt.device_count() > 0:
            return rt
    return None


def probe_direction(
    rt: GpuRuntime,
    buffer_bytes: int,
    direction: int,
    pinned: bool,
    iterations: int,
    warmup: int = 3,
) -> Dict[str, Any]:
    """Time `iterations` transfers of buffer_bytes in one direction."""
    dev = rt.malloc_device(buffer_bytes)
    if pinned:
        host = rt.malloc_host_pinned(buffer_bytes)
        host_free = rt.free_host_pinned
    else:
        raw = ctypes.create_string_buffer(buffer_bytes)
        host = ctypes.cast(raw, ctypes.c_void_p)
        host_free = lambda _p: None  # noqa: E731 -- freed by refcount on `raw`

    try:
        if direction == MEMCPY_HOST_TO_DEVICE:
            dst, src = dev, host
        else:
            dst, src = host, dev

        for _ in range(warmup):
            rt.memcpy(dst, src, buffer_bytes, direction)
        rt.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            rt.memcpy(dst, src, buffer_bytes, direction)
        rt.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        rt.free_device(dev)
        host_free(host)

    total = buffer_bytes * iterations
    return {
        "buffer_size_mb": buffer_bytes // (1024 * 1024),
        "direction": "h2d" if direction == MEMCPY_HOST_TO_DEVICE else "d2h",
        "memory": "pinned" if pinned else "pageable",
        "iterations": iterations,
        "bytes_transferred": total,
        "elapsed_seconds": elapsed,
        "gbps": (total / elapsed) / 1e9 if elapsed > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="results/hardware/pcie.json")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--pageable", action="store_true", help="Also measure pageable (non-pinned) host memory"
    )
    args = parser.parse_args()

    rt = load_runtime()
    if rt is None:
        print(
            "[-] No HIP or CUDA runtime with a visible device was found.\n"
            "    This probe measures real DMA over PCIe and has no simulation mode, so\n"
            "    there is nothing it can report from this machine. Run it on the target\n"
            "    box (Phase 0 is marked [MEASURE] for exactly this reason).",
            file=sys.stderr,
        )
        return 2

    print(f"[+] Using {rt.kind.upper()} runtime, {rt.device_count()} device(s) visible.")

    sizes_mb = [1, 4, 16, 64]
    modes: List[bool] = [True] + ([False] if args.pageable else [])
    results = []

    for pinned in modes:
        for direction in (MEMCPY_HOST_TO_DEVICE, MEMCPY_DEVICE_TO_HOST):
            for sz in sizes_mb:
                try:
                    res = probe_direction(
                        rt, sz * 1024 * 1024, direction, pinned, args.iterations
                    )
                except RuntimeError as e:
                    print(f"[-] {sz} MB {'pinned' if pinned else 'pageable'}: {e}", file=sys.stderr)
                    return 1
                print(
                    f"  {res['memory']:<8} {res['direction']} {sz:3d} MB: {res['gbps']:6.2f} GB/s"
                )
                results.append(res)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({"runtime": rt.kind, "results": results}, f, indent=2)

    print(f"\nSaved PCIe probe report to {out_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
