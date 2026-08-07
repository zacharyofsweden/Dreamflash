"""DeepSeek-V4 Flash Interactive Terminal Chat Interface."""

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
    print(" DREAMFLASH DEEPSEEK-V4 FLASH INTERACTIVE CHAT")
    print("==================================================================")
    print(f"[+] Model File : {MODEL_PATH}")
    print(f"[+] File Size  : {MODEL_PATH.stat().st_size / (1024**3):.2f} GB")
    print("[+] Initializing llama_cpp backend...")

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
        n_gpu_layers=0,
        verbose=False,
    )
    t_load = time.perf_counter() - t0
    print(f"[+] Model loaded successfully in {t_load:.2f} seconds!")
    print("==================================================================")
    print("Type your prompt below (or 'exit' / 'quit' to stop):\n")

    while True:
        try:
            prompt = input("User > ")
            if prompt.strip().lower() in ["exit", "quit", "q"]:
                print("Exiting chat. Goodbye!")
                break

            if not prompt.strip():
                continue

            print("\nAssistant > ", end="", flush=True)
            t_start = time.perf_counter()

            # Stream response tokens live to terminal
            stream = llm(
                f"<｜User｜>{prompt}<｜Assistant｜>",
                max_tokens=256,
                stop=["<｜end of sentence｜>", "<｜User｜>"],
                stream=True,
                echo=False,
            )

            token_count = 0
            for chunk in stream:
                token_text = chunk["choices"][0]["text"]
                print(token_text, end="", flush=True)
                token_count += 1

            t_elapsed = time.perf_counter() - t_start
            tok_s = token_count / t_elapsed if t_elapsed > 0 else 0
            print(f"\n\n[Stats: {token_count} tokens generated in {t_elapsed:.2f}s ({tok_s:.2f} tok/s)]\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break


if __name__ == "__main__":
    main()
