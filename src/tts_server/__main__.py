"""Entry point: `python -m tts_server`."""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from tts_server.settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="tts-server")
    parser.add_argument("--config", help="Path to TOML config file")
    parser.add_argument("--host", help="Override host from config")
    parser.add_argument("--port", type=int, help="Override port from config")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    # uvicorn re-imports the app factory in its worker, so propagate --config
    # via env-var; the factory will read TTS_CONFIG_FILE if no explicit path
    # was passed.
    if args.config:
        os.environ["TTS_CONFIG_FILE"] = args.config

    settings = load_settings(args.config)
    host = args.host or settings.server.host
    port = args.port or settings.server.port

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    uvicorn.run(
        "tts_server.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
