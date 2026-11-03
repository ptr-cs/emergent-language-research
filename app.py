#!/usr/bin/env python3
"""Run the application locally; no installation or publication is needed."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import threading
import webbrowser

from emergent_lab.config import DEFAULT_GENERATION_TIMEOUT, validate_config, validate_url
from emergent_lab.server import LabApp


def main():
    parser = argparse.ArgumentParser(description="Local Ollama self-conversation and two-computer communication lab.")
    parser.add_argument("--port", type=int, default=8765, help="Loopback dashboard port (default: 8765)")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="This computer's Ollama URL")
    parser.add_argument("--model", default="gpt-oss:20b", help="Downloaded model used by this computer (default: gpt-oss:20b)")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--peer-listen", metavar="ADDRESS", help="Enable peer server, e.g. 0.0.0.0 for LAN or 127.0.0.1 for an SSH tunnel")
    parser.add_argument("--peer-port", type=int, default=8766, help="Authenticated peer port (default: 8766)")
    parser.add_argument("--generation-timeout", type=int, default=DEFAULT_GENERATION_TIMEOUT,
                        help="Maximum seconds for one model reply (default: 7200 / 2 hours)")
    parser.add_argument("--open", action="store_true", help="Open the local dashboard in your browser")
    args = parser.parse_args()
    for port in (args.port, args.peer_port):
        if not 1 <= port <= 65535:
            parser.error("Ports must be between 1 and 65535.")
    if not 10 <= args.generation_timeout <= 7200:
        parser.error("Generation timeout must be between 10 and 7200 seconds.")
    try:
        validate_config({"model": args.model})
        validate_url(args.ollama_url)
        app = LabApp(args.data_dir, args.model, args.ollama_url, args.port,
                     args.peer_listen, args.peer_port, args.generation_timeout)
        app.start_servers()
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Could not start the lab: {exc}\n")
    stop = threading.Event()

    def on_signal(signum, frame):
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    print(f"Emergent Language Lab — http://127.0.0.1:{app.port}", flush=True)
    print(f"Model: {args.model} | Ollama: {args.ollama_url}", flush=True)
    print(f"Runs saved to: {args.data_dir.resolve()}", flush=True)
    if args.peer_listen:
        print(f"Peer listener: {args.peer_listen}:{app.peer_port} (token available in the local dashboard)", flush=True)
    print("Press Ctrl+C to stop. No cloud services are used.", flush=True)
    if args.open:
        webbrowser.open(f"http://127.0.0.1:{app.port}")
    try:
        stop.wait()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
