#!/usr/bin/env python3
"""Repository quality gates for complexity, duplication, and documentation."""

import argparse
import ast
import collections
import io
import json
import pathlib
import sys
import tokenize
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tools" / "quality_gates_config.json"
DEFAULT_BASELINE = ROOT / "tools" / "quality_gates_baseline.json"


class Finding(NamedTuple):
    """A single quality gate violation."""

    rule: str
    path: str
    line: int
    name: str
    message: str

    @property
    def key(self) -> str:
        """Stable key used by the ratchet baseline."""
        return "{}:{}:{}".format(self.rule, self.path, self.name)

    def to_dict(self) -> dict:
        """Convert the finding to a JSON-serializable dictionary."""
        return {
            "rule": self.rule,
            "path": self.path,
            "line": self.line,
            "name": self.name,
            "message": self.message,
            "key": self.key,
        }


class FunctionComplexity(ast.NodeVisitor):
    """Compute a simple McCabe-style cyclomatic complexity score."""

    DECISION_NODES = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.ExceptHandler,
        ast.IfExp,
        ast.comprehension,
    )

    def __init__(self) -> None:
        """Initialize the complexity score."""
        self.complexity = 1

    def generic_visit(self, node: ast.AST) -> None:
        """Count branching nodes while traversing a function body."""
        if isinstance(node, self.DECISION_NODES):
            self.complexity += 1
        elif isinstance(node, ast.BoolOp):
            self.complexity += max(len(node.values) - 1, 0)
        super().generic_visit(node)


def load_config(path: pathlib.Path) -> dict:
    """Load quality gate configuration."""
    return json.loads(path.read_text())


def rel_path(path: pathlib.Path) -> str:
    """Return a stable repository-relative path."""
    return path.resolve().relative_to(ROOT).as_posix()


def is_excluded(path: pathlib.Path, config: dict) -> bool:
    """Check whether a file should be skipped."""
    relative = rel_path(path)
    relative_path = pathlib.PurePosixPath(relative)
    return any(relative_path.match(pattern) for pattern in config["exclude"])


def iter_python_files(config: dict) -> Iterable[pathlib.Path]:
    """Yield repository Python files covered by quality gates."""
    for include in config["include"]:
        for path in ROOT.glob(include):
            if path.is_file() and path.suffix == ".py" and not is_excluded(path, config):
                yield path


def iter_functions(tree: ast.AST) -> Iterable[Tuple[str, ast.AST]]:
    """Yield named function nodes in a module."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node


def returns_value(node: ast.AST) -> bool:
    """Return whether a function contains a non-empty return statement."""
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and child.value is not None:
            return True
    return False


def function_parameters(node: ast.AST) -> List[str]:
    """Return user-facing function parameters."""
    args = list(node.args.args) + list(node.args.kwonlyargs)
    if node.args.vararg is not None:
        args.append(node.args.vararg)
    if node.args.kwarg is not None:
        args.append(node.args.kwarg)
    return [arg.arg for arg in args if arg.arg not in ("self", "cls")]


def is_private_or_dunder(name: str) -> bool:
    """Return whether a function is private implementation detail."""
    return name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def complexity_findings(path: pathlib.Path, tree: ast.AST, config: dict) -> List[Finding]:
    """Find functions that exceed the configured complexity limit."""
    max_complexity = int(config["max_cyclomatic_complexity"])
    findings = []
    for name, node in iter_functions(tree):
        visitor = FunctionComplexity()
        for child in node.body:
            visitor.visit(child)
        if visitor.complexity > max_complexity:
            findings.append(
                Finding(
                    "complexity",
                    rel_path(path),
                    node.lineno,
                    name,
                    "cyclomatic complexity {} exceeds max {}".format(
                        visitor.complexity, max_complexity
                    ),
                )
            )
    return findings


def documentation_findings(
    path: pathlib.Path, tree: ast.AST, config: dict
) -> List[Finding]:
    """Find functions without useful functional documentation."""
    findings = []
    require_private = bool(config.get("require_private_function_docs", False))
    for name, node in iter_functions(tree):
        if is_private_or_dunder(name) and not require_private:
            continue
        doc = ast.get_docstring(node) or ""
        missing_parts = []
        if not doc.strip():
            missing_parts.append("docstring")
        if function_parameters(node) and "Args:" not in doc:
            missing_parts.append("Args section")
        if returns_value(node) and "Returns:" not in doc and "Yields:" not in doc:
            missing_parts.append("Returns/Yields section")
        if missing_parts:
            findings.append(
                Finding(
                    "documentation",
                    rel_path(path),
                    node.lineno,
                    name,
                    "missing {}".format(", ".join(missing_parts)),
                )
            )
    return findings


def parse_tree(path: pathlib.Path) -> Optional[ast.AST]:
    """Parse a Python file, returning None when syntax is invalid."""
    try:
        return ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        print("{}: syntax error: {}".format(rel_path(path), exc), file=sys.stderr)
        return None


def normalized_tokens(path: pathlib.Path) -> List[str]:
    """Tokenize a Python file into comparable tokens for duplication checks."""
    tokens = []
    source = path.read_text()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.COMMENT,
        ):
            continue
        if token.type == tokenize.STRING:
            tokens.append("STRING")
        elif token.type == tokenize.NUMBER:
            tokens.append("NUMBER")
        elif token.type == tokenize.NAME:
            tokens.append("NAME")
        else:
            tokens.append(token.string)
    return tokens


def duplication_finding(paths: Sequence[pathlib.Path], config: dict) -> List[Finding]:
    """Find repeated token windows above the configured duplication ratio."""
    window = int(config["duplication_token_window"])
    file_tokens = {rel_path(path): normalized_tokens(path) for path in paths}
    owners: Dict[Tuple[str, ...], Set[str]] = collections.defaultdict(set)
    repeated_by_file: Dict[str, Set[int]] = collections.defaultdict(set)

    for path, tokens in file_tokens.items():
        for index in range(0, max(len(tokens) - window + 1, 0)):
            owners[tuple(tokens[index : index + window])].add(path)

    repeated_windows = {key: files for key, files in owners.items() if len(files) > 1}
    for path, tokens in file_tokens.items():
        for index in range(0, max(len(tokens) - window + 1, 0)):
            if tuple(tokens[index : index + window]) in repeated_windows:
                repeated_by_file[path].update(range(index, index + window))

    total_tokens = sum(len(tokens) for tokens in file_tokens.values())
    duplicated_tokens = sum(len(indices) for indices in repeated_by_file.values())
    ratio = (duplicated_tokens / total_tokens) if total_tokens else 0.0
    max_ratio = float(config["max_duplication_percent"]) / 100.0
    if ratio <= max_ratio:
        return []
    return [
        Finding(
            "duplication",
            ".",
            0,
            "repository",
            "duplicated token ratio {:.2%} exceeds max {:.2%}".format(
                ratio, max_ratio
            ),
        )
    ]


def all_findings(config: dict) -> List[Finding]:
    """Run every quality gate and return all findings."""
    paths = sorted(set(iter_python_files(config)))
    findings = []
    for path in paths:
        tree = parse_tree(path)
        if tree is None:
            continue
        findings.extend(complexity_findings(path, tree, config))
        findings.extend(documentation_findings(path, tree, config))
    findings.extend(duplication_finding(paths, config))
    return sorted(findings, key=lambda item: (item.rule, item.path, item.line, item.name))


def load_baseline(path: pathlib.Path) -> Set[str]:
    """Load existing baseline finding keys."""
    if not path.is_file():
        return set()
    data = json.loads(path.read_text())
    return {item["key"] for item in data.get("findings", [])}


def write_baseline(path: pathlib.Path, findings: Sequence[Finding]) -> None:
    """Write the current findings as the quality ratchet baseline."""
    path.write_text(
        json.dumps(
            {
                "description": (
                    "Existing quality gate violations. New findings fail CI; "
                    "remove entries as code is refactored."
                ),
                "findings": [finding.to_dict() for finding in findings],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def print_findings(findings: Sequence[Finding]) -> None:
    """Print findings in a compact command-line format."""
    for finding in findings:
        line = "{}:{}: {} {}: {}".format(
            finding.path, finding.line, finding.rule, finding.name, finding.message
        )
        print(line)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for the quality gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline", type=pathlib.Path, default=DEFAULT_BASELINE)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    findings = all_findings(config)

    if args.update_baseline:
        write_baseline(args.baseline, findings)
        print("Wrote {} findings to {}".format(len(findings), rel_path(args.baseline)))
        return 0

    baseline = set() if args.no_baseline else load_baseline(args.baseline)
    new_findings = [finding for finding in findings if finding.key not in baseline]
    if new_findings:
        print_findings(new_findings)
        print(
            "\n{} new quality gate finding(s). Run with --update-baseline only "
            "after intentionally accepting existing debt.".format(len(new_findings))
        )
        return 1

    print(
        "Quality gates passed: {} finding(s) covered by ratchet baseline.".format(
            len(findings)
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
