"""GGUF header and tensor inspection tool for DeepSeek-V4 Flash."""

import struct
import sys
from pathlib import Path

GGUF_MAGIC = 0x46554747  # b'GGUF' in little endian


def read_gguf_metadata(file_path: Path) -> None:
    print(f"[+] Opening GGUF file: {file_path}")
    print(f"[+] File size: {file_path.stat().st_size / (1024**3):.2f} GB")

    with open(file_path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            print(f"[-] Invalid magic: {magic}")
            return

        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        print(f"[+] GGUF Version   : {version}")
        print(f"[+] Total Tensors  : {n_tensors:,}")
        print(f"[+] Metadata KVs   : {n_kv:,}")


if __name__ == "__main__":
    p = Path("D:/repos/Dreamflash/models/DeepSeek-V4-Flash-REAP-IQ2XXS-w2Q2K-AProjQ8-OutQ8-chat-v2.gguf")
    if p.exists():
        read_gguf_metadata(p)
    else:
        print(f"File not found: {p}")
