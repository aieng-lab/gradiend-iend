from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = ("gradiend", "tests", "scripts")


def _python_files():
    for root_name in SEARCH_ROOTS:
        root = ROOT / root_name
        if root.exists():
            yield from root.rglob("*.py")


def _has_future_annotations(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            return any(alias.name == "annotations" for alias in node.names)
        return False
    return False


class _Pep604AnnotationVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.lines: list[int] = []

    def _check_annotation(self, node: ast.AST | None) -> None:
        if node is not None and self._contains_bit_or(node):
            self.lines.append(node.lineno)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_annotation(node.annotation)
        if node.value is not None:
            self.visit(node.value)

    def visit_arg(self, node: ast.arg) -> None:
        self._check_annotation(node.annotation)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function(node)

    def _check_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            self.visit(arg)
        if node.args.vararg is not None:
            self.visit(node.args.vararg)
        if node.args.kwarg is not None:
            self.visit(node.args.kwarg)
        self._check_annotation(node.returns)
        for stmt in node.body:
            self.visit(stmt)

    def _contains_bit_or(self, node: ast.AST) -> bool:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return True
        return any(self._contains_bit_or(child) for child in ast.iter_child_nodes(node))


def test_pep604_annotations_use_postponed_evaluation_for_python39():
    offenders: list[str] = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        if _has_future_annotations(tree):
            continue

        visitor = _Pep604AnnotationVisitor()
        visitor.visit(tree)
        rel = path.relative_to(ROOT)
        offenders.extend(f"{rel}:{line}" for line in visitor.lines)

    assert offenders == []
