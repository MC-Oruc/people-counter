from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union
import argparse
import yaml


LineType = Tuple[int, int, int, int]

# Shared config paths and helpers
DEFAULT_CONFIG_DIR = Path.cwd() / "config"
DEFAULT_APP_CFG = DEFAULT_CONFIG_DIR / "app.yaml"
DEFAULT_LINE_CFG = DEFAULT_CONFIG_DIR / "line.yaml"

_DEFAULT_APP_YAML_TEXT = """# Application default settings
source: "0"
model: "models/yolov8n.pt"
confidence: 0.3
device: "cpu"
save_output: false
output_path: "output/output.mp4"
no_view: false
imgsz: 640
max_frames: null
line_bottom: false
line_bottom_offset: 0
line_vertical_center: false
line_vertical_offset: 0
edit_line: false
save_line_path: "config/line.yaml"
mode: "body"
"""


def ensure_default_configs() -> tuple[Path, Path]:
	"""Ensure default config files exist; return their paths.

	Returns a tuple of (app_yaml_path, line_yaml_path).
	"""
	DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
	if not DEFAULT_APP_CFG.exists():
		DEFAULT_APP_CFG.write_text(_DEFAULT_APP_YAML_TEXT, encoding="utf-8")
	if not DEFAULT_LINE_CFG.exists():
		DEFAULT_LINE_CFG.write_text("line:\n- 100\n- 200\n- 500\n- 200\n", encoding="utf-8")
	return DEFAULT_APP_CFG, DEFAULT_LINE_CFG


@dataclass
class AppConfig:
	source: Union[int, str] = 0
	line: LineType = (100, 200, 500, 200)
	lines: list[LineType] = field(default_factory=list)
	model: str = "yolov8n.pt"
	confidence: float = 0.3
	device: str = "cpu"
	save_output: bool = False
	output_path: str = "output/output.mp4"
	view: bool = True
	imgsz: int = 640
	max_frames: Optional[int] = None
	line_bottom: bool = False
	line_bottom_offset: int = 0
	line_vertical_center: bool = False
	line_vertical_offset: int = 0
	edit_line: bool = False
	save_line_path: Optional[str] = None
	mode: str = "body"  # "body" or "head"
	line_band_px: int = 0

	@staticmethod
	def parse_line(arg: str) -> LineType:
		parts = [int(p) for p in arg.split(",")]
		if len(parts) != 4:
			raise argparse.ArgumentTypeError("--line 'x1,y1,x2,y2' formatinda olmali")
		return parts[0], parts[1], parts[2], parts[3]


def _arg_to_source(arg: str) -> Union[int, str]:
	try:
		return int(arg)
	except ValueError:
		return arg


def load_config_from_yaml(path: Optional[Union[str, Path]]) -> dict:
	if not path:
		return {}
	p = Path(path)
	if not p.exists():
		# Graceful fallback: if config file doesn't exist, proceed with defaults
		# instead of aborting. This allows running without any config files.
		return {}
	with p.open("r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	return data


def build_config(argv: Optional[list[str]] = None) -> AppConfig:
	parser = argparse.ArgumentParser(
		description="Kisi sayaci: Sanal cizgi kesme ile giris/cikis sayimi"
	)
	parser.add_argument("--source", type=str, default="0", help="Kamera indeksi, video yolu veya RTSP URL")
	parser.add_argument("--line", type=AppConfig.parse_line, default=(100, 200, 500, 200), help="Cizgi: x1,y1,x2,y2")
	parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model dosyasi veya ONNX (YuNet)")
	parser.add_argument("--confidence", type=float, default=0.3, help="Tespit icin minimum guven")
	parser.add_argument("--device", type=str, default="cpu", help="Aygit: cpu/cuda (YOLO icin)")
	parser.add_argument("--save-output", action="store_true", help="Cikti videoyu kaydet")
	parser.add_argument("--output-path", type=str, default="output/output.mp4", help="Cikti video yolu")
	parser.add_argument("--no-view", action="store_true", help="Pencere gosterme")
	parser.add_argument("--imgsz", type=int, default=640, help="Girdi boyutu")
	parser.add_argument("--max-frames", type=int, default=None, help="Maksimum kare sayisi (test icin)")
	parser.add_argument("--line-bottom", action="store_true", help="Cizgiyi ekranin en altina yerlestir")
	parser.add_argument("--line-bottom-offset", type=int, default=0, help="Alttan piksel offseti")
	parser.add_argument("--line-vertical-center", action="store_true", help="Dikey cizgiyi ekran ortasina yerlestir")
	parser.add_argument("--line-vertical-offset", type=int, default=0, help="Ortadan saga(+)/sola(-) piksel offseti")
	parser.add_argument("--edit-line", action="store_true", help="Fare ile cizgi noktalarini duzenle")
	parser.add_argument("--save-line-path", type=str, default=None, help="S tusu ile kaydetmek icin YAML dosya yolu")
	parser.add_argument("--config", type=str, default=str(Path.cwd() / "config" / "app.yaml"), help="YAML konfig dosyasi (varsayilan: config/app.yaml)")
	parser.add_argument("--mode", type=str, choices=["body", "head"], default="body", help="Target mode: body/head")
	parser.add_argument("--line-band-px", type=int, default=0, help="Sayim cizgisi etrafinda ROI bant genligi (px)")

	args = parser.parse_args(argv)

	yaml_cfg = load_config_from_yaml(args.config)

	source = _arg_to_source(str(yaml_cfg.get("source", args.source)))
	line = tuple(yaml_cfg.get("line", list(args.line)))  # type: ignore
	# Parse multiple lines if provided
	lines_yaml = yaml_cfg.get("lines", None)
	lines_list: list[LineType] = []
	if isinstance(lines_yaml, list):
		for item in lines_yaml:
			if isinstance(item, (list, tuple)) and len(item) == 4:
				try:
					ln: LineType = (int(item[0]), int(item[1]), int(item[2]), int(item[3]))
					lines_list.append(ln)
				except Exception:
					pass
	model = str(yaml_cfg.get("model", args.model))
	confidence = float(yaml_cfg.get("confidence", args.confidence))
	device = str(yaml_cfg.get("device", args.device))
	save_output = bool(yaml_cfg.get("save_output", args.save_output))
	output_path = str(yaml_cfg.get("output_path", args.output_path))
	view = not bool(yaml_cfg.get("no_view", args.no_view))
	imgsz = int(yaml_cfg.get("imgsz", args.imgsz))
	max_frames = yaml_cfg.get("max_frames", args.max_frames)
	max_frames = None if max_frames in (None, "", 0, "0") else int(max_frames)
	line_bottom = bool(yaml_cfg.get("line_bottom", args.line_bottom))
	line_bottom_offset = int(yaml_cfg.get("line_bottom_offset", args.line_bottom_offset))
	line_vertical_center = bool(yaml_cfg.get("line_vertical_center", args.line_vertical_center))
	line_vertical_offset = int(yaml_cfg.get("line_vertical_offset", args.line_vertical_offset))
	edit_line = bool(yaml_cfg.get("edit_line", args.edit_line))
	save_line_path = yaml_cfg.get("save_line_path", args.save_line_path)
	save_line_path = None if save_line_path in (None, "", "none") else str(save_line_path)
	mode = str(yaml_cfg.get("mode", args.mode))
	line_band_px = int(yaml_cfg.get("line_band_px", args.line_band_px))
	# Backward compat: map legacy Turkish values to new ones
	if mode == "Vucut":
		mode = "body"
	elif mode == "Kafa":
		mode = "head"

	return AppConfig(
		source=source,
		line=(int(line[0]), int(line[1]), int(line[2]), int(line[3])),
		lines=lines_list,
		model=model,
		confidence=confidence,
		device=device,
		save_output=save_output,
		output_path=output_path,
		view=view,
		imgsz=imgsz,
		max_frames=max_frames,
		line_bottom=line_bottom,
		line_bottom_offset=line_bottom_offset,
		line_vertical_center=line_vertical_center,
		line_vertical_offset=line_vertical_offset,
		edit_line=edit_line,
        save_line_path=save_line_path,
		mode=mode,
		line_band_px=line_band_px,
	)
