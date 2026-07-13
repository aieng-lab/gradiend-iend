from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = ("tests", "gradiend")


def _python_files():
    for root_name in SEARCH_ROOTS:
        root = ROOT / root_name
        if root.exists():
            yield from root.rglob("*.py")


class _ExperimentsImportVisitor(ast.NodeVisitor):
    """Flag experiments imports in code that is selected by the CI unit filter."""

    def __init__(self, rel_path: str):
        self.rel_path = rel_path
        self.offenders: list[str] = []
        self._integration_scope_depth = 0

    def _visit_in_scope(self, node: ast.AST, is_integration: bool) -> None:
        if is_integration:
            self._integration_scope_depth += 1
        self.generic_visit(node)
        if is_integration:
            self._integration_scope_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_in_scope(node, _has_integration_mark(node.decorator_list))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_in_scope(node, _has_integration_mark(node.decorator_list))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._integration_scope_depth:
            return
        if node.module == "experiments" or (node.module and node.module.startswith("experiments.")):
            self.offenders.append(f"{self.rel_path}:{node.lineno}")

    def visit_Import(self, node: ast.Import) -> None:
        if self._integration_scope_depth:
            return
        for alias in node.names:
            if alias.name == "experiments" or alias.name.startswith("experiments."):
                self.offenders.append(f"{self.rel_path}:{node.lineno}")

    def visit_Call(self, node: ast.Call) -> None:
        if self._integration_scope_depth:
            return
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                normalized = arg.value.replace("\\", "/")
                if normalized == "experiments" or normalized.startswith("experiments/"):
                    self.offenders.append(f"{self.rel_path}:{node.lineno}")
        self.generic_visit(node)


def _attribute_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        if parent:
            return f"{parent}.{node.attr}"
    return None


def _has_integration_mark(nodes: list[ast.AST]) -> bool:
    for node in nodes:
        mark_node = node.func if isinstance(node, ast.Call) else node
        if _attribute_path(mark_node) == "pytest.mark.integration":
            return True
    return False


def _module_has_integration_mark(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets):
            continue
        value = node.value
        marks = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
        if _has_integration_mark(marks):
            return True
    return False


def test_no_imports_from_experiments_namespace():
    offenders: list[str] = []
    for path in _python_files():
        if path == Path(__file__).resolve():
            continue
        if path.relative_to(ROOT).parts[0] == "experiments":
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        if _module_has_integration_mark(tree):
            continue
        rel = str(path.relative_to(ROOT))
        visitor = _ExperimentsImportVisitor(rel)
        visitor.visit(tree)
        offenders.extend(visitor.offenders)

    assert offenders == []
