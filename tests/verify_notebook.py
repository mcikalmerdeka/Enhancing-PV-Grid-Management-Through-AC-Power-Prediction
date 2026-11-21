"""Execute a notebook's code cells in one process to verify it runs end to end.

No notebook-execution library is installed (nbclient/nbconvert are absent), so
this harness executes cell sources directly in a single namespace. It is
deliberately strict about IPython syntax: the notebook must be plain Python.

Usage:
    python tests/verify_notebook.py
    python tests/verify_notebook.py --until "STAGE 3"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Prediction of AC-Power.ipynb"

# Running a script puts its own folder on sys.path; the notebook expects the repo root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    """Read the notebook path and optional stage cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument(
        "--until",
        default=None,
        help="Stop before the markdown heading containing this text (case-insensitive).",
    )
    return parser.parse_args()


def cell_source(cell: dict) -> str:
    """Join a notebook cell's source, which JSON stores as a list of lines."""
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def main() -> int:
    """Execute cells in order and stop at the first failure."""
    args = parse_args()
    if not args.notebook.exists():
        print(f"notebook not found: {args.notebook}")
        return 2

    try:
        notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
        cells = notebook["cells"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"malformed notebook {args.notebook}: {exc}")
        return 2

    namespace = {
        "__name__": "__main__",
        "__file__": str(args.notebook),
        "display": print,
    }
    executed = 0
    started = time.perf_counter()

    for index, cell in enumerate(cells, start=1):
        source = cell_source(cell)
        kind = cell.get("cell_type")

        if kind == "markdown":
            heading = next(
                (line for line in source.splitlines() if line.lstrip().startswith("#")),
                "",
            )
            if args.until and args.until.lower() in heading.lower():
                print(f"stopping before cell {index}: {heading[:80]}")
                break
            continue

        if kind != "code" or not source.strip():
            continue

        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("%", "!")):
                print(
                    f"cell {index} uses IPython-only syntax, which this harness "
                    f"does not support: {stripped}"
                )
                return 1

        cell_started = time.perf_counter()
        try:
            exec(compile(source, f"<cell {index}>", "exec"), namespace)
        except Exception:
            print(f"\nFAILED in cell {index}:\n{source}\n", flush=True)
            traceback.print_exc()
            return 1
        finally:
            plt.close("all")

        executed += 1
        print(f"[cell {index}] {time.perf_counter() - cell_started:.1f}s")

    print(f"\nOK: {executed} code cells executed in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
