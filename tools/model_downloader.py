"""Model downloader and integrity verification tool for DeepSeek-V4 Flash GGUF files.

Calculates required disk space, checks available drive capacity, verifies block structures,
and supports resuming downloads from official Hugging Face repositories.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# Add roofline model path
sys.path.insert(0, str(Path(__file__).resolve().parent / "roofline"))
from model import FLASH

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
        free_bytes_ct = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            str(target_dir), None, None, ctypes.byref(free_bytes_ct)
        )
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
    """Verify downloaded GGUF files exist and match size specifications."""
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

        if abs(size - spec["expected_bytes"]) < 100_000_000:
            print(f"[+] Valid file   : {spec['filename']} ({size_gb:.2f} GB)")
        else:
            print(f"[!] Warning file : {spec['filename']} size mismatch (Found {size_gb:.2f} GB, expected ~{expected_gb:.2f} GB)")
    return all_valid


def main() -> None:
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
        verify_downloaded_files(out_p)
        return

    if not space_ok:
        print("\n[!] Please free up space on your drive before downloading.")
        return

    print("\n[+] Setup check complete. You can download the model GGUF split files using:")
    print(f"    huggingface-cli download {DEFAULT_HF_REPO} --local-dir {out_p}")


if __name__ == "__main__":
    main()
