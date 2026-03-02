#!/usr/bin/env python3
import argparse
import ast
import io
import re
import tokenize
from pathlib import Path
from typing import Iterable, Set, Tuple

CODING_RE = re.compile(r"^#.*coding[:=]\s*[-\w.]+")

def iter_py_files(root: Path, exclude_dirs: Set[str]) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in exclude_dirs for part in path.parts):
            continue
        if path.is_file():
            yield path

def _iter_stmt_lists(node: ast.AST):
    for field in ("body", "orelse", "finalbody"):
        value = getattr(node, field, None)
        if isinstance(value, list):
            yield value

def strip_standalone_string_exprs(src_text: str) -> Tuple[str, bool]:
    try:
        tree = ast.parse(src_text)
    except SyntaxError:
        return src_text, False

    ranges = []
    for node in ast.walk(tree):
        for stmts in _iter_stmt_lists(node):
            for stmt in stmts:
                if not isinstance(stmt, ast.Expr):
                    continue
                value = stmt.value
                if not isinstance(value, ast.Constant):
                    continue
                if not isinstance(value.value, str):
                    continue
                if not hasattr(stmt, "lineno") or not hasattr(stmt, "end_lineno"):
                    continue
                ranges.append((stmt.lineno, stmt.end_lineno))

    if not ranges:
        return src_text, False

    lines = src_text.splitlines(keepends=True)
    changed = False
    for start, end in sorted(ranges, reverse=True):
        start_idx = max(0, start - 1)
        end_idx = min(len(lines), end)
        for i in range(start_idx, end_idx):
            if lines[i] != "":
                lines[i] = "\n" if lines[i].endswith("\n") else ""
                changed = True

    if not changed:
        return src_text, False
    return "".join(lines), True

def strip_standalone_string_exprs_from_bytes(src: bytes) -> Tuple[bytes, bool]:
    try:
        text = src.decode("utf-8")
    except UnicodeDecodeError:
        text = src.decode("latin-1")

    new_text, changed = strip_standalone_string_exprs(text)
    if not changed:
        return src, False
    return new_text.encode("utf-8"), True

def compact_whitespace_bytes(src: bytes) -> Tuple[bytes, bool]:
    encoding = "utf-8"
    try:
        text = src.decode(encoding)
    except UnicodeDecodeError:
        encoding = "latin-1"
        text = src.decode(encoding)

    original = text
    lines = text.splitlines()

    compacted_lines = []
    blank_run = 0
    for line in lines:
        stripped_line = line.rstrip()
        if stripped_line == "":
            blank_run += 1
            if blank_run <= 1:
                compacted_lines.append("")
            continue

        blank_run = 0
        compacted_lines.append(stripped_line)

    while compacted_lines and compacted_lines[0] == "":
        compacted_lines.pop(0)
    while compacted_lines and compacted_lines[-1] == "":
        compacted_lines.pop()

    new_text = "\n".join(compacted_lines)
    if new_text:
        new_text += "\n"

    if new_text == original:
        return src, False
    return new_text.encode(encoding), True

def strip_comments_from_python_bytes(src: bytes) -> Tuple[bytes, bool]:
    tokens = list(tokenize.tokenize(io.BytesIO(src).readline))
    out_tokens = []
    changed = False

    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            row, _ = tok.start
            text = tok.string

            keep_shebang = row == 1 and text.startswith("#!")
            keep_coding = row <= 2 and CODING_RE.match(text) is not None

            if keep_shebang or keep_coding:
                out_tokens.append(tok)
            else:
                changed = True
            continue

        out_tokens.append(tok)

    if changed:
        src = tokenize.untokenize(out_tokens)

    src_after_strings, changed_strings = strip_standalone_string_exprs_from_bytes(src)
    src_compacted, changed_compacted = compact_whitespace_bytes(src_after_strings)
    return src_compacted, (changed or changed_strings or changed_compacted)

def process_file(path: Path, dry_run: bool) -> bool:
    src = path.read_bytes()
    try:
        new_src, changed = strip_comments_from_python_bytes(src)
    except Exception:
        new_src, changed = compact_whitespace_bytes(src)
    if changed and not dry_run:
        path.write_bytes(new_src)
    return changed

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove comments from all Python files under a directory."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Target directory (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print files that would change, without writing.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[".git", "__pycache__", ".venv", "venv", "build", "dist"],
        help="Directory name to exclude (can be repeated).",
    )

    args = parser.parse_args()
    root = Path(args.target).resolve()

    if not root.exists() or not root.is_dir():
        print(f"Invalid target directory: {root}")
        return 2

    exclude_dirs = set(args.exclude_dir)
    changed_files = []

    for py_file in iter_py_files(root, exclude_dirs):
        try:
            if process_file(py_file, args.dry_run):
                changed_files.append(py_file)
        except Exception as exc:
            print(f"[ERROR] {py_file}: {exc}")

    for file_path in changed_files:
        print(file_path)

    mode = "would be changed" if args.dry_run else "changed"
    print(f"\nDone: {len(changed_files)} file(s) {mode}.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
