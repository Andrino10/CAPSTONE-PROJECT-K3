"""Run the top-3 local retrieval evaluation and validate its results."""

from __future__ import annotations

import sys

from .check_metrics import validate
from .similarity_eval_all import print_console_report, run_evaluation


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows = run_evaluation()
    validate()
    print_console_report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
