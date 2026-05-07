#!/usr/bin/env python3
"""Convert a PowerShell-redirected reward_benchmark log into the
canonical UTF-8 single-line-per-record form that
`scripts/evaluate_gate_a.py` expects.

PowerShell's `> log.txt` redirection on Windows produces
UTF-16 LE files where each long stdout line is HARD-WRAPPED
at ~120 columns AND PowerShell's stderr-error-wrapping
(NativeCommandError + Category/FullyQualified noise) is
interleaved into the file.  Linux/macOS shells do none of
this, so logs from those platforms are pass-through.

Usage:
    python3 scripts/clean_powershell_log.py <input.log> <output.log>

This is idempotent; running on an already-clean log is a
near-noop.
"""

import io
import re
import sys
from pathlib import Path


# Lines we drop wholesale — PowerShell error wrapping noise that
# appears around cargo's stderr output.
DROP_RE = re.compile(
    r"^("
    r"In Zeile:|"
    r"\+ +~+$|"
    r"\+ +Category|"
    r"\+ +Fully|"
    r"cargo +:"
    r")",
)
DROP_RE_INDENTED = re.compile(
    r"^ {2,}(\+ Category|\+ Fully)",
)

# A new logical record starts on lines beginning with one of these.
RECORD_PREFIXES = (
    "[iter-",
    "Iter-",
    "Aggregate",
    "Verdict",
    "| Seed",
    "| ---",
    "Smoke",
    "_n_seeds",
    "###",
    "Running",
    "cargo",
    "=== ",
    "   ",  # cargo's leading-spaces output
)
# Tables of the form "| 42 | 0.0312 | ..."
TABLE_ROW_RE = re.compile(r"^\| +[0-9]+ +\|")


def _is_record_start(line: str) -> bool:
    if line.startswith(RECORD_PREFIXES):
        return True
    if TABLE_ROW_RE.match(line):
        return True
    return False


def clean(src_path: Path, dst_path: Path) -> None:
    # Detect encoding by sniffing the first two bytes for UTF-16 BOM
    raw = src_path.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        text = raw.decode("utf-16-le").lstrip("﻿")
    elif raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16-be").lstrip("﻿")
    else:
        text = raw.decode("utf-8", errors="replace")

    out_lines = []
    buf = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if DROP_RE.match(line) or DROP_RE_INDENTED.match(line):
            continue
        if line.strip() == "":
            if buf:
                out_lines.append(buf)
                buf = ""
            out_lines.append("")
            continue
        if _is_record_start(line):
            if buf:
                out_lines.append(buf)
            buf = line
        else:
            buf = buf + " " + line.lstrip() if buf else line
    if buf:
        out_lines.append(buf)

    # Collapse multi-space runs (PowerShell wrap-and-indent injects them)
    # but leave the table cell separators intact (they use ` | `).
    cleaned = []
    space_re = re.compile(r"[ \t]+")
    for line in out_lines:
        cleaned.append(space_re.sub(" ", line))

    dst_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    src = Path(argv[0])
    dst = Path(argv[1])
    clean(src, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
