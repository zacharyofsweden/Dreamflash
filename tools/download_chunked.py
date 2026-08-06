"""Chunked GGUF downloader with resume support and progress reporting."""

import os
import sys
import time
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/jabbatheduck/DeepSeek-v4-flash-mini/resolve/main/DeepSeek-V4-Flash-REAP-IQ2XXS-w2Q2K-AProjQ8-OutQ8-chat-v2.gguf"
TARGET_DIR = Path("D:/repos/Dreamflash/models")
TARGET_FILE = TARGET_DIR / "DeepSeek-V4-Flash-REAP-IQ2XXS-w2Q2K-AProjQ8-OutQ8-chat-v2.gguf"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks


def download_with_resume() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    file_mode = "ab"
    downloaded_bytes = 0

    if TARGET_FILE.exists():
        downloaded_bytes = TARGET_FILE.stat().st_size
        print(f"[+] Found existing partial file: {downloaded_bytes / (1024**3):.2f} GB downloaded.")

    req = urllib.request.Request(URL)
    if downloaded_bytes > 0:
        req.add_header("Range", f"bytes={downloaded_bytes}-")

    print(f"[+] Connecting to Hugging Face...")
    try:
        with urllib.request.urlopen(req) as resp:
            content_length = resp.headers.get("Content-Length")
            total_bytes = downloaded_bytes + (int(content_length) if content_length else 0)
            total_gb = total_bytes / (1024**3)

            print(f"[+] Total Model Size : {total_gb:.2f} GB")
            print(f"[+] Starting download loop to {TARGET_FILE}...")

            start_time = time.perf_counter()
            bytes_since_start = 0

            with open(TARGET_FILE, file_mode) as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
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

            print(f"\n[+] Download finished successfully!")
    except Exception as e:
        print(f"\n[-] Download paused or interrupted: {e}")


if __name__ == "__main__":
    download_with_resume()
