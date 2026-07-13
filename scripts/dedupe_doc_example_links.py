"""Remove duplicate [:material-file-code-outline: ...] links per example file in each doc."""
from __future__ import annotations

import re
from pathlib import Path

PATTERN = re.compile(
    r"^\[:material-file-code-outline: `([^`]+)`\]\([^)]+\)\s*\n",
    re.MULTILINE,
)

SKIP = {"examples.md"}


def dedupe_file(path: Path) -> int:
    if path.name in SKIP:
        return 0
    text = path.read_text(encoding="utf-8")
    seen: set[str] = set()
    removed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal removed
        name = match.group(1)
        if name in seen:
            removed += 1
            return ""
        seen.add(name)
        return match.group(0)

    new_text = PATTERN.sub(repl, text)
    new_text = re.sub(r"\n{3,}", "\n\n", new_text)
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
    return removed


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "docs"
    total = 0
    for path in sorted(root.rglob("*.md")):
        n = dedupe_file(path)
        if n:
            print(f"{path.relative_to(root.parent)}: removed {n}")
            total += n
    print(f"Total removed: {total}")


if __name__ == "__main__":
    main()
