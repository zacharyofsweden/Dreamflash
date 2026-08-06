"""Pre-download disk-space check and post-download size check for DS4-Flash GGUF files.

This tool does NOT download anything -- it prints the huggingface-cli command to
run. It does not verify checksums or parse GGUF block structures either; the
"verify" step is a file-size comparison only. For an actual chunked download with
validated resume, use tools/download_chunked.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Default Hugging Face repository and file manifest for DeepSeek-V4 Flash GGUF
DEFAULT_HF_REPO = "antirez/deepseek-v4-gguf"
MODEL_FILES = {
    "flash_gguf": {
        "filename": "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf",
        "expected_bytes": 86_720_111_488,
        "description": "DeepSeek-V4-Flash IQ2_XXS / Q2_K mixed quant GGUF (ds4 native)",
    },
}


def check_disk_space(target_dir: Path, required_bytes: int) -> bool:
    """Verify target directory has sufficient free disk space."""
    target_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "statvfs"):
        # Unix / Linux
        stat = os.statvfs(str(target_dir))
        free_bytes = stat.f_bavail * stat.f_frsize
    else:
        # Windows
        import ctypes
        from ctypes import wintypes

        free_bytes_ct = ctypes.c_ulonglong(0)
        fn = ctypes.windll.kernel32.GetDiskFreeSpaceExW
        fn.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
        ]
        fn.restype = wintypes.BOOL
        if not fn(str(target_dir), None, None, ctypes.byref(free_bytes_ct)):
            err = ctypes.get_last_error()
            print(
                f"[-] ERROR: GetDiskFreeSpaceExW failed for {target_dir} (error {err}); "
                f"cannot determine free space.",
                file=sys.stderr,
            )
            return False
        free_bytes = free_bytes_ct.value

    free_gb = free_bytes / (1024**3)
    required_gb = required_bytes / (1024**3)

    print(f"[+] Target Drive Space : {free_gb:.2f} GB free")
    print(f"[+] Required Space     : {required_gb:.2f} GB")

    if free_bytes < required_bytes:
        print(f"[-] ERROR: Insufficient disk space! Need {required_gb:.2f} GB but only {free_gb:.2f} GB available.", file=sys.stderr)
        return False
    return True


def verify_downloaded_files(output_dir: Path) -> bool:
    """Check that the GGUF files exist and are the expected size.

    This is a SIZE check, not an integrity check. There is no checksum in
    MODEL_FILES to verify against, so this cannot detect a file that is the
    right length but wrong content. Verify the published checksum by hand
    before trusting a download.
    """
    all_valid = True
    print("\n=== Verifying Model Files ===")
    for key, spec in MODEL_FILES.items():
        file_path = output_dir / spec["filename"]
        if not file_path.exists():
            print(f"[-] Missing file : {spec['filename']} ({spec['description']})")
            all_valid = False
            continue

        size = file_path.stat().st_size
        size_gb = size / (1024**3)
        expected_gb = spec["expected_bytes"] / (1024**3)

        if size == spec["expected_bytes"]:
            print(f"[+] Correct size : {spec['filename']} ({size_gb:.2f} GB)")
        else:
            delta = size - spec["expected_bytes"]
            print(
                f"[-] SIZE MISMATCH: {spec['filename']} is {size:,} bytes, "
                f"expected {spec['expected_bytes']:,} ({delta:+,}); "
                f"{size_gb:.2f} GB vs {expected_gb:.2f} GB"
            )
            all_valid = False
    return all_valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to store model weights")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing files without downloading")
    args = parser.parse_args()

    out_p = Path(args.output_dir).resolve()
    total_required_bytes = sum(s["expected_bytes"] for s in MODEL_FILES.values())

    print("==================================================================")
    print(" DREAMFLASH MODEL CHECKPOINT MANAGER")
    print("==================================================================")

    space_ok = check_disk_space(out_p, total_required_bytes)

    if args.verify_only:
        return 0 if verify_downloaded_files(out_p) else 1

    if not space_ok:
        print("\n[!] Please free up space on your drive before downloading.")
        return 1

    print("\n[+] Setup check complete. This tool does not download; run either:")
    print(f"    huggingface-cli download {DEFAULT_HF_REPO} --local-dir {out_p}")
    print(f"    python tools/download_chunked.py --target-dir {out_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
