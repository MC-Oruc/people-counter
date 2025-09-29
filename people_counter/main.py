from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import cv2
import numpy as np

from .config import build_config
from .core.engine import PeopleEngine
from .utils.line_io import load_line_for_source, save_line_for_source
from queue import Queue, Empty
import threading
import time
from .utils.capture import open_capture
@dataclass
class _DraggableLine:
	x1: int
	y1: int
	x2: int
	y2: int
	grab_radius: int = 12
	active_handle: Optional[str] = None  # 'p1', 'p2', or None

	def as_tuple(self) -> Tuple[int, int, int, int]:
		return self.x1, self.y1, self.x2, self.y2

	def draw(self, frame: np.ndarray) -> np.ndarray:
		cv2.line(frame, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 255), 2)
		cv2.circle(frame, (self.x1, self.y1), self.grab_radius, (0, 255, 0), 2)
		cv2.circle(frame, (self.x2, self.y2), self.grab_radius, (0, 0, 255), 2)
		return frame

	def _hit(self, x: int, y: int, px: int, py: int) -> bool:
		return (x - px) ** 2 + (y - py) ** 2 <= self.grab_radius ** 2

	def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
		if event == cv2.EVENT_LBUTTONDOWN:
			if self._hit(self.x1, self.y1, x, y):
				self.active_handle = 'p1'
			elif self._hit(self.x2, self.y2, x, y):
				self.active_handle = 'p2'
		elif event == cv2.EVENT_MOUSEMOVE and self.active_handle is not None:
			if self.active_handle == 'p1':
				self.x1, self.y1 = x, y
			elif self.active_handle == 'p2':
				self.x2, self.y2 = x, y
		elif event == cv2.EVENT_LBUTTONUP:
			self.active_handle = None


def _overlay_help(frame: np.ndarray, text: str) -> np.ndarray:
	overlay = frame.copy()
	cv2.rectangle(overlay, (10, 10), (10 + 520, 60), (0, 0, 0), -1)
	cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
	cv2.putText(frame, text, (16, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
	return frame


def interactive_edit(window_name: str, frame: np.ndarray, line_xyxy: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
	line = _DraggableLine(*line_xyxy)
	cv2.setMouseCallback(window_name, line.on_mouse)
	text = "Noktayı sürükle: Sol tık. Kaydet: S. İptal: ESC/Q"
	while True:
		disp = frame.copy()
		disp = line.draw(disp)
		disp = _overlay_help(disp, text)
		cv2.imshow(window_name, disp)
		key = cv2.waitKey(1) & 0xFF
		if key in (27, ord('q')):
			return line_xyxy  # cancel returns original
		if key in (ord('s'), ord('S')):
			return line.as_tuple()


def ensure_writer(path: str, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	return cv2.VideoWriter(path, fourcc, fps, frame_size)


def main(argv: List[str] | None = None) -> int:
	cfg = build_config(argv)

	cap = open_capture(cfg.source)
	# Reduce buffering for live sources (best-effort; optional)
	try:
		if isinstance(cfg.source, int) or (isinstance(cfg.source, str) and cfg.source.isdigit()):
			cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	except Exception:
		pass
	if not cap.isOpened():
		raise RuntimeError(f"Kaynak açılamadı: {cfg.source}")

	fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# Determine line placement (single) unless multiple lines provided in config
	line_to_use = cfg.line
	if not getattr(cfg, 'lines', []):
		if cfg.line_bottom:
			y = max(0, min(height - 1, height - 1 - int(cfg.line_bottom_offset)))
			line_to_use = (0, y, max(0, width - 1), y)
		elif cfg.line_vertical_center:
			x = max(0, min(width - 1, (width // 2) + int(cfg.line_vertical_offset)))
			line_to_use = (x, 0, x, max(0, height - 1))

	# Optional interactive edit step before starting loop
	window_name = "Kişi Sayacı"
	if cfg.edit_line:
		ret, preview = cap.read()
		if ret:
			cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
			edited = interactive_edit(window_name, preview, line_to_use)
			line_to_use = edited
			if cfg.save_line_path:
				save_line_for_source(cfg.save_line_path, cfg.source, edited)
			# reset stream to start
			cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
	else:
		# If no interactive edit and a per-source line exists at save_line_path, load it
		try:
			if getattr(cfg, 'save_line_path', None):
				stored = load_line_for_source(cfg.save_line_path, cfg.source)
				if stored:
					line_to_use = stored
		except Exception:
			pass

	# Use shared engine
	engine = PeopleEngine()
	engine.set_mode(cfg.mode)
	if getattr(cfg, 'lines', []):
		try:
			engine.set_lines([(int(l[0]), int(l[1]), int(l[2]), int(l[3])) for l in cfg.lines])
		except Exception:
			engine.set_line(line_to_use)
	else:
		engine.set_line(line_to_use)
	# Apply optional ROI band
	try:
		engine.set_line_band(getattr(cfg, 'line_band_px', 0))
	except Exception:
		pass
	engine.init_detector(cfg.model, cfg.confidence, cfg.imgsz)
	# Ensure device is applied if detector supports it
	try:
		if hasattr(engine.detector, 'set_device'):
			engine.detector.set_device(cfg.device)
	except Exception:
		pass

	writer = None
	if cfg.save_output:
		writer = ensure_writer(cfg.output_path, fps, (width, height))

	# Threaded pipeline: capture thread -> queue -> inference on main thread
	q: "Queue[np.ndarray]" = Queue(maxsize=2)
	stop_event = threading.Event()
	producer_done = {"flag": False}

	def _capture_worker() -> None:
		try:
			while not stop_event.is_set():
				ok, fr = cap.read()
				if not ok:
					break
				try:
					q.put(fr, timeout=0.1)
				except Exception:
					# queue full; drop frame
					pass
		except Exception:
			pass
		finally:
			producer_done["flag"] = True

	thr = threading.Thread(target=_capture_worker, daemon=True)
	thr.start()

	frame_idx = 0
	last_log = time.time()
	proc_frames = 0
	while True:
		try:
			frame = q.get(timeout=0.5)
		except Empty:
			if producer_done["flag"]:
				break
			else:
				continue

		frame = engine.process(frame)

		if cfg.view:
			cv2.imshow(window_name, frame)
			key = cv2.waitKey(1) & 0xFF
			if key in (27, ord("q")):
				break

		if writer is not None:
			writer.write(frame)

		frame_idx += 1
		proc_frames += 1
		if cfg.max_frames is not None and frame_idx >= cfg.max_frames:
			break
		# simple telemetry: print every ~1s
		now = time.time()
		if (now - last_log) >= 1.0:
			fps_now = proc_frames / max(1e-6, (now - last_log))
			print(f"[headless] fps={fps_now:.1f} q={q.qsize()}")
			proc_frames = 0
			last_log = now

	# shutdown
	stop_event.set()
	try:
		thr.join(timeout=1.0)
	except Exception:
		pass
	cap.release()
	if writer is not None:
		writer.release()
	cv2.destroyAllWindows()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
