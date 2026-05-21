#!/usr/bin/env python3
"""Strict repository lint rules not covered by the formatter."""

import argparse
import ast
import pathlib
import sys
import tokenize
from typing import Iterable, List, NamedTuple, Optional, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAX_LINE_LENGTH = 90
EXCLUDED_PARTS = {".git", "__pycache__", "build", "dist"}
EXCLUDED_PREFIXES = ("bazel-",)
CHECKED_DIRS = ("hawk_eye", "tools")


class Finding(NamedTuple):
    """A strict lint finding."""

    path: str
    line: int
    code: str
    message: str

    def format(self) -> str:
        """Format the finding for command-line output.

        Returns:
            Human-readable finding text.
        """
        return f"{self.path}:{self.line}: {self.code} {self.message}"


class PublicFunctionVisitor(ast.NodeVisitor):
    """Find public functions that are missing return annotations."""

    def __init__(self, path: pathlib.Path) -> None:
        """Initialize the visitor.

        Args:
            path: File being inspected.
        """
        self.path = path
        self.findings: List[Finding] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Inspect a function definition.

        Args:
            node: Function definition AST node.
        """
        self._check_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Inspect an async function definition.

        Args:
            node: Async function definition AST node.
        """
        self._check_function(node)
        self.generic_visit(node)

    def _check_function(self, node: ast.AST) -> None:
        if node.name.startswith("_"):
            return
        if node.returns is not None:
            return
        self.findings.append(
            Finding(
                rel_path(self.path),
                node.lineno,
                "ANN001",
                "missing return annotation",
            )
        )


def rel_path(path: pathlib.Path) -> str:
    """Return a repository-relative path.

    Args:
        path: Path to normalize.

    Returns:
        Repository-relative POSIX path.
    """
    return path.resolve().relative_to(ROOT).as_posix()


def is_excluded(path: pathlib.Path) -> bool:
    """Return whether a path should be excluded.

    Args:
        path: Path to inspect.

    Returns:
        Whether the path is excluded from strict linting.
    """
    relative_parts = set(path.relative_to(ROOT).parts)
    if relative_parts.intersection(EXCLUDED_PARTS):
        return True
    return any(part.startswith(EXCLUDED_PREFIXES) for part in path.parts)


def iter_python_files() -> Iterable[pathlib.Path]:
    """Yield Python files covered by strict linting.

    Yields:
        Python source paths.
    """
    for checked_dir in CHECKED_DIRS:
        for path in (ROOT / checked_dir).rglob("*.py"):
            if not is_excluded(path):
                yield path


def line_findings(path: pathlib.Path) -> List[Finding]:
    """Run strict physical-line checks.

    Args:
        path: File to inspect.

    Returns:
        Line-level lint findings.
    """
    findings = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        findings.extend(_line_findings(path, line_number, line))
    return findings


def _line_findings(path: pathlib.Path, line_number: int, line: str) -> List[Finding]:
    findings = []
    if "\t" in line:
        findings.append(Finding(rel_path(path), line_number, "LNT001", "tab indentation"))
    if len(line) > MAX_LINE_LENGTH:
        findings.append(
            Finding(rel_path(path), line_number, "LNT002", "line longer than 90 chars")
        )
    if line.rstrip() != line:
        findings.append(
            Finding(rel_path(path), line_number, "LNT003", "trailing whitespace")
        )
    return findings


def syntax_findings(path: pathlib.Path) -> List[Finding]:
    """Run AST-based strict checks.

    Args:
        path: File to inspect.

    Returns:
        Syntax-level lint findings.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    visitor = PublicFunctionVisitor(path)
    visitor.visit(tree)
    return visitor.findings


def token_findings(path: pathlib.Path) -> List[Finding]:
    """Run token-based strict checks.

    Args:
        path: File to inspect.

    Returns:
        Token-level lint findings.
    """
    findings = []
    source_lines = path.read_text().splitlines()
    with tokenize.open(path) as file_obj:
        for token in tokenize.generate_tokens(file_obj.readline):
            if token.start[0] > len(source_lines):
                continue
            source_line = source_lines[token.start[0] - 1]
            findings.extend(_token_findings(path, token, source_line))
    return findings


def _token_findings(
    path: pathlib.Path, token: tokenize.TokenInfo, source_line: str
) -> List[Finding]:
    if token.type != tokenize.STRING:
        return []
    if "f\"" in source_line or "f'" in source_line:
        return []
    if token.string.startswith(("'", "'''")):
        return [Finding(rel_path(path), token.start[0], "STR001", "prefer double quotes")]
    return []


def file_findings(path: pathlib.Path) -> List[Finding]:
    """Run every strict lint check for one file.

    Args:
        path: File to inspect.

    Returns:
        Strict lint findings for the file.
    """
    return line_findings(path) + syntax_findings(path) + token_findings(path)


def all_findings() -> List[Finding]:
    """Run strict lint checks for the repository.

    Returns:
        All strict lint findings.
    """
    findings = []
    for path in sorted(iter_python_files()):
        findings.extend(file_findings(path))
    return findings


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    findings = all_findings()
    for finding in findings:
        print(finding.format())
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
