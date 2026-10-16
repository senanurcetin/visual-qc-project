"""Dependency-free CircleCI config and notebook preflight, no network, no execution."""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CIRCLECI_FILE = ROOT / ".circleci" / "config.yml"
NOTEBOOKS_DIR = ROOT / "notebooks"
RAW_HEREDOC = re.compile(r"(?<!\\)<<(['\"]?)[A-Za-z_][A-Za-z0-9_]*\1")
SECRET_PATTERNS = (
    re.compile(r"KGAT_[a-zA-Z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
    re.compile(r"AIza[0-9a-zA-Z_-]{30,}"),
)


def check_circleci(source: str) -> None:
    for lineno, line in enumerate(source.splitlines(), 1):
        if RAW_HEREDOC.search(line) and not line.lstrip().startswith("#"):
            raise ValueError(
                f".circleci/config.yml:{lineno}: unescaped heredoc syntax; escape the less-than signs"
            )


def check_notebook(path: Path) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name}: invalid notebook JSON") from exc
    if data.get("nbformat") != 4 or not isinstance(data.get("cells"), list):
        raise ValueError(f"{path.name}: expected nbformat 4 with cells")
    for index, cell in enumerate(data["cells"], 1):
        if not isinstance(cell, dict) or cell.get("cell_type") not in ("code", "markdown", "raw"):
            raise ValueError(f"{path.name}: invalid cell {index}")
        source = cell.get("source", [])
        if not isinstance(source, (list, str)):
            raise ValueError(f"{path.name}: invalid source in cell {index}")
        source_text = "".join(source) if isinstance(source, list) else source
        for pattern in SECRET_PATTERNS:
            if pattern.search(source_text):
                raise ValueError(f"{path.name}: possible embedded credential in cell {index}")
        if cell["cell_type"] == "code":
            if cell.get("outputs"):
                raise ValueError(f"{path.name}: saved output in cell {index}")
            if cell.get("execution_count") is not None:
                raise ValueError(f"{path.name}: saved execution count in cell {index}")
            try:
                ast.parse(source_text, filename=f"{path.name}:cell-{index}")
            except SyntaxError as exc:
                raise ValueError(
                    f"{path.name}: code cell {index} has Python syntax error, line {exc.lineno}"
                ) from exc


def main() -> int:
    if not CIRCLECI_FILE.is_file():
        print("FAIL: missing CircleCI config", file=sys.stderr)
        return 1
    try:
        check_circleci(CIRCLECI_FILE.read_text(encoding="utf-8"))
        notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
        for notebook in notebooks:
            check_notebook(notebook)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: CircleCI heredoc guard and {len(notebooks)} notebook source(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
