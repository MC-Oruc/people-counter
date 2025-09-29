from __future__ import annotations

from pathlib import Path
from typing import Tuple
import yaml


LineType = Tuple[int, int, int, int]
LinesType = list[LineType]


# --- Source key helpers -----------------------------------------------------
def _norm_path_str(p: Path) -> str:
    try:
        # Prefer relative within cwd for stability
        rel = p.resolve().relative_to(Path.cwd().resolve())
        return rel.as_posix()
    except Exception:
        try:
            return p.resolve().as_posix()
        except Exception:
            return str(p).replace("\\", "/")


def _source_to_key(source: int | str) -> str:
    """Normalize a capture source to a stable key for per-source storage.

    camera:<index> for integer sources
    rtsp:<url> for RTSP URLs
    video:<path> for file paths (relative to cwd when possible)
    other:<text> for anything else
    """
    try:
        # Local import to avoid heavy cv2 import if not needed at module import time
        from .capture import _to_source, is_rtsp_url  # type: ignore
    except Exception:
        _to_source = lambda v: v  # type: ignore
        def is_rtsp_url(_v):  # type: ignore
            return False

    src = _to_source(source)
    if isinstance(src, int):
        return f"camera:{src}"
    s = str(src).strip()
    if is_rtsp_url(s):
        return f"rtsp:{s}"
    # Treat as a path-like
    try:
        p = Path(s)
        return f"video:{_norm_path_str(p)}"
    except Exception:
        return f"other:{s}"


def _parse_lines_node(node) -> LinesType:
    out: LinesType = []
    if isinstance(node, list):
        # Could be list of 4-int or list of lists
        if len(node) == 4 and all(isinstance(x, (int, float)) for x in node):
            out.append((int(node[0]), int(node[1]), int(node[2]), int(node[3])))
        else:
            for item in node:
                if isinstance(item, (list, tuple)) and len(item) == 4:
                    out.append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
    elif isinstance(node, dict):
        # support {line: [...]} or {lines: [[...], ...]}
        if "lines" in node:
            return _parse_lines_node(node.get("lines"))
        if "line" in node:
            return _parse_lines_node(node.get("line"))
    return out


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


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
    _ensure_parent(p)
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


# --- Per-source aware helpers ----------------------------------------------
def load_lines_for_source(path: str | Path, source: int | str, default: LinesType | None = None) -> LinesType:
    p = Path(path)
    if default is None:
        default = []
    try:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # Try per_source block first
            key = _source_to_key(source)
            per = data.get("per_source")
            if isinstance(per, dict) and key in per:
                out = _parse_lines_node(per.get(key))
                if out:
                    return out
            # Fallback to global schema
            out = _parse_lines_node(data.get("lines"))
            if out:
                return out
            out = _parse_lines_node(data.get("line"))
            if out:
                return out
        # File not found or nothing valid: fallback
    except Exception:
        pass
    return list(default)


def save_lines_for_source(path: str | Path, source: int | str, lines: LinesType) -> None:
    p = Path(path)
    _ensure_parent(p)
    try:
        current = {}
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                current = yaml.safe_load(f) or {}
        if not isinstance(current, dict):
            current = {}
        key = _source_to_key(source)
        per = current.get("per_source")
        if not isinstance(per, dict):
            per = {}
        # Store consistently using 'lines'
        per[key] = {"lines": [list(map(int, ln)) for ln in lines]}
        current["per_source"] = per
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(current, f, allow_unicode=True, sort_keys=True)
    except Exception:
        # Best-effort fallback to global writer
        save_lines(p, lines)


def load_line_for_source(path: str | Path, source: int | str, default: LineType = (100, 200, 500, 200)) -> LineType:
    lst = load_lines_for_source(path, source, default=[default])
    return lst[0] if lst else default


def save_line_for_source(path: str | Path, source: int | str, line: LineType) -> None:
    # Store single line under per_source, using single-line schema for readability
    p = Path(path)
    _ensure_parent(p)
    try:
        current = {}
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                current = yaml.safe_load(f) or {}
        if not isinstance(current, dict):
            current = {}
        key = _source_to_key(source)
        per = current.get("per_source")
        if not isinstance(per, dict):
            per = {}
        per[key] = {"line": list(map(int, line))}
        current["per_source"] = per
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(current, f, allow_unicode=True, sort_keys=True)
    except Exception:
        save_line(p, line)
