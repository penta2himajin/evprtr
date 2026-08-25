"""CLI entry: python -m compositor"""

from __future__ import annotations

import os

import uvicorn

from compositor.api import build_app


def main() -> None:
    host = os.environ.get("EVPRTR_HOST", "127.0.0.1")
    port = int(os.environ.get("EVPRTR_PORT", "8741"))
    app = build_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
