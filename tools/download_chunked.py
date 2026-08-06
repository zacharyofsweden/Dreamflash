"""Chunked GGUF downloader with validated resume support and progress reporting.

Resume is only safe if the server actually honours the Range request. A server
that ignores `Range` answers 200 with the *whole* body, and appending that to a
partial file produces a silently corrupted result that is larger than the real
file. Hugging Face redirects to a CDN, so this is a live failure mode, not a
hypothetical one. This tool therefore refuses to append unless it sees a 206
with a Content-Range whose start offset matches what is already on disk.
"""

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://huggingface.co/jabbatheduck/DeepSeek-v4-flash-mini/resolve/main/"
    "DeepSeek-V4-Flash-REAP-IQ2XXS-w2Q2K-AProjQ8-OutQ8-chat-v2.gguf"
)
DEFAULT_FILENAME = "DeepSeek-V4-Flash-REAP-IQ2XXS-w2Q2K-AProjQ8-OutQ8-chat-v2.gguf"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks


def _parse_content_range_start(header: str) -> int:
    """Parse the start offset out of a `Content-Range: bytes START-END/TOTAL` header.

    Returns -1 if the header is absent or unparseable.
    """
    if not header:
        return -1
    try:
        unit, _, rng = header.strip().partition(" ")
        if unit.lower() != "bytes":
            return -1
        span, _, _total = rng.partition("/")
        start, _, _end = span.partition("-")
        return int(start)
    except (ValueError, AttributeError):
        return -1


def download_with_resume(url: str, target_file: Path, chunk_size: int = CHUNK_SIZE) -> int:
    """Download `url` to `target_file`, resuming a partial file when safe.

    Returns 0 on a verified-complete download, non-zero otherwise.
    """
    target_file.parent.mkdir(parents=True, exist_ok=True)

    downloaded_bytes = target_file.stat().st_size if target_file.exists() else 0
    if downloaded_bytes > 0:
        print(f"[+] Found existing partial file: {downloaded_bytes / (1024**3):.2f} GB on disk.")
        print("[!] Resume trusts the on-disk bytes. If they came from a different URL or a")
        print("    different revision of this file, delete it and start over.")

    req = urllib.request.Request(url)
    if downloaded_bytes > 0:
        req.add_header("Range", f"bytes={downloaded_bytes}-")

    print("[+] Connecting to Hugging Face...")
    try:
        with urllib.request.urlopen(req) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            content_length = resp.headers.get("Content-Length")
            remaining = int(content_length) if content_length else 0

            if downloaded_bytes > 0:
                # We asked for a range. Anything other than a 206 starting at exactly
                # our offset means appending would corrupt the file.
                if status != 206:
                    print(
                        f"\n[-] ABORT: requested a byte range but the server answered "
                        f"{status}, not 206.\n"
                        f"    The response body is the ENTIRE file; appending it to the "
                        f"{downloaded_bytes / (1024**3):.2f} GB already on disk would "
                        f"silently corrupt it.\n"
                        f"    Delete {target_file} and re-run to download from scratch."
                    )
                    return 2

                start = _parse_content_range_start(resp.headers.get("Content-Range", ""))
                if start != downloaded_bytes:
                    print(
                        f"\n[-] ABORT: server returned 206 but Content-Range starts at "
                        f"{start}, not {downloaded_bytes}.\n"
                        f"    Refusing to append at the wrong offset. Delete "
                        f"{target_file} and re-run."
                    )
                    return 2

                total_bytes = downloaded_bytes + remaining
            else:
                total_bytes = remaining

            total_gb = total_bytes / (1024**3)
            print(f"[+] Total Model Size : {total_gb:.2f} GB")
            print(f"[+] Starting download loop to {target_file}...")

            start_time = time.perf_counter()
            bytes_since_start = 0

            with open(target_file, "ab") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    bytes_since_start += len(chunk)

                    elapsed = time.perf_counter() - start_time
                    mbps = (bytes_since_start / elapsed) / 1e6 if elapsed > 0 else 0
                    pct = (downloaded_bytes / total_bytes) * 100 if total_bytes > 0 else 0

                    print(
                        f"\rDownloading: {downloaded_bytes / (1024**3):6.2f} / {total_gb:.2f} GB "
                        f"({pct:5.1f}%) | Speed: {mbps:6.1f} MB/s",
                        end="",
                        flush=True,
                    )

                # Make the on-disk length reflect everything we wrote, so that an
                # interrupted run resumes from the right offset next time.
                f.flush()
                os.fsync(f.fileno())
    except KeyboardInterrupt:
        print("\n[-] Interrupted by user. Re-run to resume.")
        return 130
    except Exception as e:
        print(f"\n[-] Download FAILED (not merely paused): {e}")
        print("    Re-run to resume; the partial file is left in place.")
        return 1

    final_size = target_file.stat().st_size
    if total_bytes > 0 and final_size != total_bytes:
        print(
            f"\n[-] INCOMPLETE: expected {total_bytes:,} bytes, have {final_size:,}. "
            f"Re-run to resume."
        )
        return 1

    print(f"\n[+] Download finished: {final_size:,} bytes.")
    print("[!] Size is not integrity. Verify the checksum against the model card before use.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Source URL")
    parser.add_argument(
        "--target-dir",
        default=os.environ.get("DREAMFLASH_MODEL_DIR", "models"),
        help="Destination directory (default: ./models, or $DREAMFLASH_MODEL_DIR)",
    )
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="Destination filename")
    parser.add_argument(
        "--chunk-size-mb", type=int, default=8, help="Read chunk size in MB (default: 8)"
    )
    args = parser.parse_args()

    target_file = Path(args.target_dir).expanduser() / args.filename
    return download_with_resume(args.url, target_file, args.chunk_size_mb * 1024 * 1024)


if __name__ == "__main__":
    sys.exit(main())
