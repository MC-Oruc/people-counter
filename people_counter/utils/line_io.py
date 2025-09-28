from __future__ import annotations

from pathlib import Path
from typing import Tuple
import yaml


LineType = Tuple[int, int, int, int]
LinesType = list[LineType]


def load_lines(path: str | Path, default: LinesType | None = None) -> LinesType:
    p = Path(path)
    if default is None:
        default = []
    try:
        if not p.exists():
            return list(default)
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Prefer new schema: list of lines
        lines = data.get("lines")
        out: LinesType = []
        if isinstance(lines, list):
            for item in lines:
                if isinstance(item, (list, tuple)) and len(item) == 4:
                    out.append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
        if out:
            return out
        # Backward compat: single 'line' key
        line = data.get("line")
        if isinstance(line, (list, tuple)) and len(line) == 4:
            return [(int(line[0]), int(line[1]), int(line[2]), int(line[3]))]
    except Exception:
        pass
    return list(default)


def save_lines(path: str | Path, lines: LinesType) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable = [list(map(int, ln)) for ln in lines]
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"lines": serializable}, f, allow_unicode=True)


def load_line(path: str | Path, default: LineType = (100, 200, 500, 200)) -> LineType:
    # Backward compatibility: return the first line if multiple are present
    lines = load_lines(path, default=[default])
    return lines[0] if lines else default


def save_line(path: str | Path, line: LineType) -> None:
    # Backward compatibility: write a single line file
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"line": list(line)}, f, allow_unicode=True)
