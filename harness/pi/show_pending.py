#!/usr/bin/env python3
"""Print pending Path-B approvals in a form easy for a supervisor agent to review.

Usage::

    python harness/pi/show_pending.py --state-dir /path/to/.evprtr/pi-rpc

Exit codes: 0 = one or more pending, 1 = none, 2 = error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args()
    state = Path(args.state_dir).resolve()
    pending_dir = state / "pending"
    if not pending_dir.is_dir():
        print(json.dumps({"pending": [], "state_dir": str(state)}))
        return 1
    items: list[dict] = []
    for path in sorted(pending_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            items.append({"id": path.stem, "error": str(exc), "path": str(path)})
            continue
        items.append(
            {
                "id": data.get("id") or path.stem,
                "path": str(path),
                "toolName": data.get("toolName") or data.get("tool_name"),
                "message": data.get("message") or data.get("confirmMessage"),
                "raw": data,
            }
        )
    print(json.dumps({"pending": items, "count": len(items), "state_dir": str(state)}, indent=2))
    return 0 if items else 1


if __name__ == "__main__":
    sys.exit(main())
