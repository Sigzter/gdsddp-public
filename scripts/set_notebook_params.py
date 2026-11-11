#!/usr/bin/env python3
"""Set parameters in the first code cell that assigns them in a notebook, before executing it.

    python3 scripts/set_notebook_params.py notebooks/04_run_gdsddp.ipynb T=102 K=200 'H_VALUES=[0.5, 1.0]' RUN_NAME='"full"'
    python3 scripts/set_notebook_params.py notebooks/04_run_gdsddp.ipynb --file cluster/nb_params_full.txt

Each argument NAME=VALUE (or each line of the --file) replaces the line `NAME = ...` (keeping its trailing comment) in the
parameter cell of the notebook. VALUE is inserted verbatim as Python source. The notebook is
written back in place; execute it afterwards with jupyter nbconvert --execute --inplace.
"""

import re
import sys
from pathlib import Path

import nbformat


def main(path: str, settings: list[str]) -> int:
    nb = nbformat.read(path, as_version=4)
    pairs = [s.split("=", 1) for s in settings]
    done = {k: False for k, _ in pairs}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        lines = cell.source.split("\n")
        for i, line in enumerate(lines):
            for k, v in pairs:
                m = re.match(rf"^{re.escape(k)}\s*=\s*[^#]*?(\s*#.*)?$", line)
                if m and not line.lstrip().startswith("#"):
                    lines[i] = f"{k} = {v}" + (m.group(1) or ""); done[k] = True
        cell.source = "\n".join(lines)
    missing = [k for k, ok in done.items() if not ok]
    if missing:
        print("not found:", missing); return 1
    nbformat.write(nb, path)
    print("set", ", ".join(f"{k} = {v}" for k, v in pairs), "in", path)
    return 0


if __name__ == "__main__":
    args = sys.argv[2:]
    if args[:1] == ["--file"]:
        args = [ln.strip() for ln in Path(args[1]).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    raise SystemExit(main(sys.argv[1], args))
