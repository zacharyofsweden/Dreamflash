"""Dreamflash Web Server & Streaming Chat UI for DeepSeek-V4 Flash.

Binds strictly to 127.0.0.1:8080.
Serves a responsive, dark-mode real-time streaming web chat interface with live token speed counters.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MODEL_PATH = Path("D:/repos/Dreamflash/models/DeepSeek-V4-Flash-REAP-IQ2XXS-w2Q2K-AProjQ8-OutQ8-chat-v2.gguf")

# Global LLM instance initialized on server startup
llm_instance = None


def init_model():
    global llm_instance
    if llm_instance is not None:
        return llm_instance

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    print(f"[+] Loading model into memory: {MODEL_PATH} ({MODEL_PATH.stat().st_size / (1024**3):.2f} GB)...")
    from llama_cpp import Llama

    t0 = time.perf_counter()
    llm_instance = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,
        n_threads=os.cpu_count() or 8,
        n_gpu_layers=0,
        verbose=False,
    )
    t_load = time.perf_counter() - t0
    print(f"[+] Model successfully loaded in {t_load:.2f} seconds!")
    return llm_instance


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dreamflash - DeepSeek V4 Flash Web Chat</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f8fafc;
            --accent-color: #38bdf8;
            --accent-hover: #0284c7;
            --user-msg-bg: #0369a1;
            --assistant-msg-bg: #334155;
            --border-color: #475569;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }

        header {
            background-color: var(--card-bg);
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        h1 {
            margin: 0;
            font-size: 1.3rem;
            color: var(--accent-color);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .stats-bar {
            display: flex;
            gap: 1.5rem;
            font-size: 0.9rem;
        }

        .stat-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }

        .stat-value {
            font-weight: bold;
            color: var(--accent-color);
        }

        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem 2rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .message {
            max-width: 80%;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .user-message {
            align-self: flex-end;
            background-color: var(--user-msg-bg);
            color: #fff;
            border-bottom-right-radius: 2px;
        }

        .assistant-message {
            align-self: flex-start;
            background-color: var(--assistant-msg-bg);
            border-bottom-left-radius: 2px;
            border: 1px solid var(--border-color);
        }

        #input-container {
            padding: 1rem 2rem;
            background-color: var(--card-bg);
            border-top: 1px solid var(--border-color);
            display: flex;
            gap: 1rem;
        }

        textarea {
            flex: 1;
            background-color: var(--bg-color);
            color: var(--text-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.8rem;
            font-size: 1rem;
            resize: none;
            height: 24px;
            outline: none;
        }

        textarea:focus {
            border-color: var(--accent-color);
        }

        button {
            background-color: var(--accent-color);
            color: #000;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 0 1.5rem;
            font-size: 1rem;
            cursor: pointer;
            transition: background-color 0.2s;
        }

        button:hover {
            background-color: var(--accent-hover);
        }

        button:disabled {
            background-color: var(--border-color);
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <header>
        <h1>⚡ Dreamflash: DeepSeek-V4 Flash Engine</h1>
        <div class="stats-bar">
            <div class="stat-item">Model: <span class="stat-value">DS4-Flash 53.4GB</span></div>
            <div class="stat-item">Live Speed: <span id="speed-meter" class="stat-value">0.0 tok/s</span></div>
            <div class="stat-item">Tokens: <span id="token-counter" class="stat-value">0</span></div>
        </div>
    </header>

    <div id="chat-container">
        <div class="message assistant-message">Hello! I am DeepSeek-V4-Flash running locally via the Dreamflash streaming engine. Ask me anything to test generation speed!</div>
    </div>

    <div id="input-container">
        <textarea id="user-input" placeholder="Type your message... (Press Enter to send)" rows="1"></textarea>
        <button id="send-btn">Send</button>
    </div>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const speedMeter = document.getElementById('speed-meter');
        const tokenCounter = document.getElementById('token-counter');

        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        sendBtn.addEventListener('click', sendMessage);

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;

            // Add user message
            appendMessage(text, 'user-message');
            userInput.value = '';
            userInput.disabled = true;
            sendBtn.disabled = true;

            // Create assistant placeholder message
            const assistantMsgEl = appendMessage('', 'assistant-message');

            let tokens = 0;
            const startTime = performance.now();

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: text })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value, { stream: true });
                    const lines = chunk.split('\\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.replace('data: ', '').trim();
                            if (dataStr === '[DONE]') break;

                            try {
                                const data = JSON.parse(dataStr);
                                if (data.token) {
                                    assistantMsgEl.textContent += data.token;
                                    tokens++;
                                    const elapsedSec = (performance.now() - startTime) / 1000;
                                    const tokPerSec = (tokens / elapsedSec).toFixed(1);
                                    speedMeter.textContent = `${tokPerSec} tok/s`;
                                    tokenCounter.textContent = tokens;
                                    chatContainer.scrollTop = chatContainer.scrollHeight;
                                }
                            } catch (err) {}
                        }
                    }
                }
            } catch (err) {
                assistantMsgEl.textContent += '\\n[Error generating response: ' + err.message + ']';
            } finally {
                userInput.disabled = false;
                sendBtn.disabled = false;
                userInput.focus();
            }
        }

        function appendMessage(text, className) {
            const msgEl = document.createElement('div');
            msgEl.className = `message ${className}`;
            msgEl.textContent = text;
            chatContainer.appendChild(msgEl);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return msgEl;
        }
    </script>
</body>
</html>
"""


class DreamflashHTTPRequestHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Override to suppress default HTTP access logs for clean SSE output
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(HTML_PAGE.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                data = json.loads(body)
                prompt = data.get("prompt", "")
            except Exception:
                self.send_error(400, "Invalid JSON payload")
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            llm = init_model()

            formatted_prompt = f"<｜User｜>{prompt}<｜Assistant｜>"
            stream = llm(
                formatted_prompt,
                max_tokens=256,
                stop=["<｜end of sentence｜>", "<｜User｜>"],
                stream=True,
                echo=False,
            )

            for chunk in stream:
                token_text = chunk["choices"][0]["text"]
                sse_data = f"data: {json.dumps({'token': token_text})}\n\n"
                try:
                    self.wfile.write(sse_data.encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    break

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            self.send_error(404, "API endpoint not found")


def main() -> None:
    host = "127.0.0.1"  # Bound strictly to 127.0.0.1
    port = 8080

    print("==================================================================")
    print(" DREAMFLASH STREAMING WEB CHAT SERVER")
    print("==================================================================")
    print(f"[+] Binding strictly to: http://{host}:{port}")
    print("[+] Loading model weights in background...")

    try:
        init_model()
    except Exception as e:
        print(f"[-] Failed to load model: {e}", file=sys.stderr)
        return

    server = HTTPServer((host, port), DreamflashHTTPRequestHandler)
    print(f"\n[+] SERVER READY! Open http://{host}:{port} in your browser to talk to DeepSeek-V4 Flash.")
    print("[+] Press Ctrl+C to stop server.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Server stopped cleanly.")
        server.server_close()


if __name__ == "__main__":
    main()
