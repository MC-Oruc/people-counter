from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
import argparse

from . import __version__
from .utils.line_io import (
	load_line_for_source,
	save_line_for_source,
)
from .config import ensure_default_configs, DEFAULT_APP_CFG, DEFAULT_LINE_CFG
from .utils.capture import open_capture


DEFAULT_CONFIG_DIR = Path.cwd() / "config"


def _ensure_config_files() -> None:
	ensure_default_configs()


# removed: legacy helper _read_line_from_file


def cmd_run(argv_rest: list[str]) -> int:
	# Forward remaining args to library main
	from . import main as app_main
	return app_main.main(argv_rest)


def cmd_ui(argv_rest: list[str]) -> int:
	# Launch Qt UI
	from .ui import qt_app
	return qt_app.main()


def cmd_edit_line(args: argparse.Namespace) -> int:
	import cv2
	from .main import interactive_edit
	source_text = str(args.source)
	try:
		source: str | int = int(source_text)
	except ValueError:
		source = source_text
	path = Path(args.output)
	path.parent.mkdir(parents=True, exist_ok=True)
	cap = open_capture(source)
	if not cap.isOpened():
		print(f"Kaynak acilamadi: {source}")
		return 1
	ret, frame = cap.read()
	cap.release()
	if not ret:
		print("Kare okunamadi")
		return 1
	cv2.namedWindow("Cizgi Duzenle", cv2.WINDOW_NORMAL)
	# Load per-source saved line if any, fallback to old global schema
	init_line = load_line_for_source(path, source)
	edited = interactive_edit("Cizgi Duzenle", frame, init_line)
	try:
		# Save under per-source key
		save_line_for_source(path, source, edited)
	except Exception as e:
		print(f"Kaydetme basarisiz: {e}")
		return 1
	print(f"Cizgi kaydedildi: {path}")
	return 0


def cmd_config_init() -> int:
	_ensure_config_files()
	print(f"Olusturuldu/var: {DEFAULT_APP_CFG}")
	print(f"Olusturuldu/var: {DEFAULT_LINE_CFG}")
	return 0


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		prog="people-counter",
		description="Kisi sayaci CLI",
	)
	p.add_argument("--version", action="version", version=f"people-counter {__version__}")
	sub = p.add_subparsers(dest="cmd", required=True)

	prun = sub.add_parser("run", help="Baslat (headless)")
	prun.add_argument("args", nargs=argparse.REMAINDER, help="Uygulamaya aktarilacak argumanlar")

	pui = sub.add_parser("ui", help="Qt arayuzunu baslat")
	# No extra args for now

	pedit = sub.add_parser("edit-line", help="Cizgiyi duzenle ve kaydet")
	pedit.add_argument("--source", type=str, default="0", help="Kamera indeksi, video yolu veya RTSP URL")
	pedit.add_argument("--output", type=str, default=str(DEFAULT_LINE_CFG), help="Cizgi YAML cikti yolu")

	pcfg = sub.add_parser("config-init", help="Varsayilan config dosyalarini olustur")

	pst = sub.add_parser("self-test", help="Hizli oz-test: 10 kare CPU/GPU + UI ac/kapat")

	return p


def main(argv: Optional[list[str]] = None) -> int:
	argv = list(sys.argv[1:] if argv is None else argv)
	parser = build_parser()
	args = parser.parse_args(argv)
	if args.cmd == "run":
		return cmd_run(args.args)
	if args.cmd == "ui":
		return cmd_ui([])
	if args.cmd == "edit-line":
		return cmd_edit_line(args)
	if args.cmd == "config-init":
		return cmd_config_init()
	if args.cmd == "self-test":
		from .selftest import run as selftest_run
		return selftest_run()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
