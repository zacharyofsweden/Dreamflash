"""Live DeepSeek-V4 Flash GGUF Inference Test Runner."""

import os
import sys
import time
from pathlib import Path

MODEL_PATH = Path("D:/repos/Dreamflash/models/DeepSeek-V4-Flash-REAP-IQ2XXS-w2Q2K-AProjQ8-OutQ8-chat-v2.gguf")


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"[-] ERROR: Model file not found at {MODEL_PATH}", file=sys.stderr)
        return

    print("==================================================================")
    print(" DREAMFLASH LIVE DEEPSEEK-V4 FLASH INFERENCE TEST")
    print("==================================================================")
    print(f"[+] Loading GGUF Model : {MODEL_PATH}")
    print(f"[+] Size               : {MODEL_PATH.stat().st_size / (1024**3):.2f} GB")

    try:
        from llama_cpp import Llama
    except ImportError:
        print("[-] ERROR: llama_cpp module not installed.", file=sys.stderr)
        return

    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,
        n_threads=os.cpu_count() or 8,
        n_gpu_layers=0,  # 0 for CPU streaming, higher for VRAM offloading
        verbose=True,
    )
    t_load = time.perf_counter() - t0
    print(f"\n[+] Model Loaded in {t_load:.2f} seconds!")

    prompt = "Explain quantum computing in three simple sentences:"
    print(f"\n[+] Prompt: '{prompt}'")
    print("\n[+] Generating completion...\n")

    t_gen_start = time.perf_counter()
    output = llm(
        prompt,
        max_tokens=128,
        stop=["\n\n", "User:"],
        echo=False,
    )
    t_gen_elapsed = time.perf_counter() - t_gen_start

    response_text = output["choices"][0]["text"]
    usage = output.get("usage", {})
    gen_tokens = usage.get("completion_tokens", 0)
    tok_s = gen_tokens / t_gen_elapsed if t_gen_elapsed > 0 else 0

    print("--- MODEL OUTPUT ---")
    print(response_text.strip())
    print("--------------------")
    print(f"[+] Generated {gen_tokens} tokens in {t_gen_elapsed:.2f}s ({tok_s:.2f} tok/s)")
    print("==================================================================")


if __name__ == "__main__":
    main()
