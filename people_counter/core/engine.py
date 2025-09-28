from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Tuple, Dict, Optional
import numpy as np
import supervision as sv
import cv2

from ..detector import build_detector

@dataclass
class Counts:
	incoming: int = 0
	outgoing: int = 0


class LineCounter:
	def __init__(self, line: Tuple[int, int, int, int], method: str = "linezone") -> None:
		x1, y1, x2, y2 = line
		self._coords: Tuple[int, int, int, int] = (x1, y1, x2, y2)
		self.line = sv.LineZone(sv.Point(x1, y1), sv.Point(x2, y2))
		self.counts = Counts()
		# counting method: 'linezone' (default), 'centroid', 'bottom', 'bbox'
		self.method: str = method
		# per-tracker state for custom methods
		self._tid_last_side: Dict[int, int] = {}
		self._tid_last_point: Dict[int, Tuple[float, float]] = {}
		self._tid_last_seen: Dict[int, int] = {}
		self._tid_last_span: Dict[int, Tuple[int, int]] = {}
		self._tick: int = 0

	def set_line(self, line: Tuple[int, int, int, int]) -> None:
		prev_in = self.counts.incoming
		prev_out = self.counts.outgoing
		x1, y1, x2, y2 = line
		self._coords = (x1, y1, x2, y2)
		self.line = sv.LineZone(sv.Point(x1, y1), sv.Point(x2, y2))
		# preserve counts in our own object (LineZone counts reset when line changes)
		self.counts.incoming = prev_in
		self.counts.outgoing = prev_out
		# reset per-tracker side cache (geometry changed)
		self._tid_last_side.clear()
		self._tid_last_point.clear()
		self._tid_last_seen.clear()
		self._tid_last_span.clear()
		self._tick = 0

	def set_method(self, method: str) -> None:
		# Switch counting method; preserve cumulative counts
		allowed = {"linezone", "centroid", "bottom", "bbox"}
		self.method = method if method in allowed else "linezone"
		# Reset only per-tracker caches to avoid inconsistent transitions
		self._tid_last_side.clear()
		self._tid_last_point.clear()
		self._tid_last_seen.clear()
		self._tid_last_span.clear()
		self._tick = 0

	def reset(self) -> None:
		self.counts.incoming = 0
		self.counts.outgoing = 0
		x1, y1, x2, y2 = self._coords
		self.line = sv.LineZone(sv.Point(x1, y1), sv.Point(x2, y2))
		self._tid_last_side.clear()
		self._tid_last_point.clear()
		self._tid_last_seen.clear()
		self._tick = 0

	def update(self, detections: sv.Detections) -> None:
		# Dispatch to selected counting strategy
		if self.method == "linezone":
			self.line.trigger(detections)
			self.counts.incoming = int(self.line.in_count)
			self.counts.outgoing = int(self.line.out_count)
			return
		# Custom methods below
		self._tick += 1
		# Line geometry and orientation
		x1, y1, x2, y2 = self._coords
		p1 = (float(x1), float(y1))
		p2 = (float(x2), float(y2))
		dx = p2[0] - p1[0]
		dy = p2[1] - p1[1]
		# Helper: cross product sign of point relative to directed line p1->p2
		def side(px: float, py: float) -> int:
			cp = dx * (py - p1[1]) - dy * (px - p1[0])
			if cp > 1.0:
				return 1
			if cp < -1.0:
				return -1
			return 0
		# Helper: check if two segments (a1-a2) and (b1-b2) intersect
		def _seg_intersect(a1: Tuple[float, float], a2: Tuple[float, float], b1: Tuple[float, float], b2: Tuple[float, float]) -> bool:
			def _orient(p: Tuple[float,float], q: Tuple[float,float], r: Tuple[float,float]) -> float:
				return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
			def _on_seg(p: Tuple[float,float], q: Tuple[float,float], r: Tuple[float,float]) -> bool:
				return min(p[0], r[0]) - 1e-6 <= q[0] <= max(p[0], r[0]) + 1e-6 and min(p[1], r[1]) - 1e-6 <= q[1] <= max(p[1], r[1]) + 1e-6
			o1 = _orient(a1, a2, b1)
			o2 = _orient(a1, a2, b2)
			o3 = _orient(b1, b2, a1)
			o4 = _orient(b1, b2, a2)
			if (o1 == 0 and _on_seg(a1, b1, a2)) or (o2 == 0 and _on_seg(a1, b2, a2)) or (o3 == 0 and _on_seg(b1, a1, b2)) or (o4 == 0 and _on_seg(b1, a2, b2)):
				return True
			return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)
		# Extract ids and boxes
		ids = detections.tracker_id if detections.tracker_id is not None else []
		xyxy = detections.xyxy if hasattr(detections, 'xyxy') else None
		seen_now: set[int] = set()
		for i, tid in enumerate(ids):
			if tid is None or xyxy is None:
				continue
			itid = int(tid)
			seen_now.add(itid)
			xa, ya, xb, yb = map(float, xyxy[i])
			cx = 0.5 * (xa + xb)
			cy = 0.5 * (ya + yb)
			if self.method == "bottom":
				px, py = cx, yb
			elif self.method == "centroid":
				px, py = cx, cy
			elif self.method == "bbox":
				# For bbox method, we consider the whole rectangle side relative to line
				px, py = cx, cy
			else:
				px, py = cx, cy
			s_curr = side(px, py)
			p_curr = (px, py)
			p_prev: Optional[Tuple[float,float]] = self._tid_last_point.get(itid)
			s_prev: Optional[int] = self._tid_last_side.get(itid)
			span_prev: Optional[Tuple[int,int]] = self._tid_last_span.get(itid)
			# store current seen tick
			self._tid_last_seen[itid] = self._tick
			if p_prev is None or s_prev is None:
				self._tid_last_point[itid] = p_curr
				self._tid_last_side[itid] = s_curr
				if self.method == "bbox":
					# initialize span
					corners = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
					signs = [side(px_, py_) for (px_, py_) in corners]
					smin, smax = (min(signs), max(signs))
					self._tid_last_span[itid] = (smin, smax)
				continue
			crossed = False
			if self.method == "bbox":
				# Require entire bbox to be on one side then on the other side
				corners = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
				signs = [side(px_, py_) for (px_, py_) in corners]
				smin, smax = (min(signs), max(signs))
				pure_prev = span_prev is not None and span_prev[0] == span_prev[1] and span_prev[0] != 0
				pure_curr = (smin == smax) and (smin != 0)
				if pure_prev and pure_curr and span_prev[0] != smin:
					# Decide direction based on sign change
					if span_prev[0] < smin:
						self.counts.incoming += 1
					else:
						self.counts.outgoing += 1
				# update caches for bbox
				self._tid_last_span[itid] = (smin, smax)
				self._tid_last_point[itid] = p_curr
				self._tid_last_side[itid] = s_curr
				continue
			# centroid or bottom methods
			if s_prev != 0 and s_curr != 0 and s_prev != s_curr:
				# ensure segment intersects the finite line segment to avoid far-side flips
				if _seg_intersect(p_prev, p_curr, p1, p2):
					crossed = True
			if crossed:
				# Direction: negative->positive treated as incoming; positive->negative as outgoing
				if s_prev < s_curr:
					self.counts.incoming += 1
				else:
					self.counts.outgoing += 1
			# update caches
			self._tid_last_point[itid] = p_curr
			self._tid_last_side[itid] = s_curr
		# prune stale trackers occasionally
		if (self._tick % 30) == 0:
			ttl = 180  # ~ few seconds depending on FPS
			old = [tid for tid, last in self._tid_last_seen.items() if (self._tick - last) > ttl]
			for tid in old:
				self._tid_last_seen.pop(tid, None)
				self._tid_last_point.pop(tid, None)
				self._tid_last_side.pop(tid, None)
				self._tid_last_span.pop(tid, None)

	def draw(self, frame: np.ndarray) -> np.ndarray:
		# Draw only the line without "in/out" labels on it; counts are shown separately
		x1, y1, x2, y2 = self._coords
		cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
		return frame


class Visualizer:
	def __init__(self) -> None:
		self.bbox_annotator = sv.BoxAnnotator(
			thickness=2,
		)
		self.label_annotator = sv.LabelAnnotator(
			text_thickness=1,
			text_scale=0.5,
		)
		# trails per tracker id
		self._trail_points_by_tid: dict[int, deque[tuple[int, int]]] = {}
		self._trail_last_seen: dict[int, int] = {}
		self._frame_index: int = 0
		self.trail_max_len: int = 64
		self.trail_ttl_frames: int = 120
		self.show_trails: bool = True

	def draw_detections(self, frame: np.ndarray, detections: sv.Detections, labels: list[str]) -> np.ndarray:
		frame = self.bbox_annotator.annotate(scene=frame, detections=detections)
		frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)
		return frame

	def update_trails(self, detections: sv.Detections) -> None:
		self._frame_index += 1
		ids = detections.tracker_id if detections.tracker_id is not None else []
		xyxy = detections.xyxy if hasattr(detections, 'xyxy') else None
		for idx, tid in enumerate(ids):
			if tid is None or xyxy is None:
				continue
			x1, y1, x2, y2 = map(int, xyxy[idx])
			cx = (x1 + x2) // 2
			cy = (y1 + y2) // 2
			trail = self._trail_points_by_tid.get(int(tid))
			if trail is None:
				trail = deque(maxlen=self.trail_max_len)
				self._trail_points_by_tid[int(tid)] = trail
			trail.append((cx, cy))
			self._trail_last_seen[int(tid)] = self._frame_index
		# prune old trails
		to_delete = [tid for tid, last in self._trail_last_seen.items() if (self._frame_index - last) > self.trail_ttl_frames]
		for tid in to_delete:
			self._trail_last_seen.pop(tid, None)
			self._trail_points_by_tid.pop(tid, None)

	def draw_trails(self, frame: np.ndarray) -> np.ndarray:
		if not self.show_trails:
			return frame
		# draw behind boxes
		for tid, pts in self._trail_points_by_tid.items():
			if len(pts) < 2:
				continue
			# simple polyline with fading thickness
			points = list(pts)
			for i in range(1, len(points)):
				p1 = points[i - 1]
				p2 = points[i]
				th = max(1, 3 - (len(points) - i) // 8)
				cv2.line(frame, p1, p2, (0, 165, 255), th)
		return frame

	def draw_roi_band(self, frame: np.ndarray, line_xyxy: Tuple[int, int, int, int], band_px: int) -> np.ndarray:
		if band_px <= 0:
			return frame
		x1, y1, x2, y2 = line_xyxy
		h, w = frame.shape[:2]
		xmin = max(0, min(x1, x2) - band_px)
		xmax = min(w - 1, max(x1, x2) + band_px)
		ymin = max(0, min(y1, y2) - band_px)
		ymax = min(h - 1, max(y1, y2) + band_px)
		vis = frame.copy()
		cv2.rectangle(vis, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (60, 60, 60), -1)
		cv2.addWeighted(vis, 0.25, frame, 0.75, 0, frame)
		return frame

	def draw_counts(self, frame: np.ndarray, incoming: int, outgoing: int) -> np.ndarray:
		overlay = frame.copy()
		text = f"Giris: {incoming} | Cikis: {outgoing}"
		cv2.rectangle(overlay, (10, 10), (260, 40), (0, 0, 0), -1)
		cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
		cv2.putText(frame, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
		return frame


@dataclass
class EngineState:
	# Deprecated: no longer used; retained for compatibility if imported elsewhere
	mode: str = "body"
	line: Tuple[int, int, int, int] = (100, 200, 500, 200)
	confidence: float = 0.3
	imgsz: int = 640
	model_path: str = "yolov8n.pt"


class PeopleEngine:
	def __init__(self) -> None:
		self.detector = None
		self.tracker = sv.ByteTrack()
		# Support multiple counting lines
		self._line_counters: list[LineCounter] = [LineCounter((100, 200, 500, 200))]
		self.viz = Visualizer()
		self.mode: str = "body"
		# ROI band around the lines: disabled by default
		self._roi_band_px: int = 0  # 0 disables
		# Counting method
		self._count_method: str = "linezone"

	def init_detector(self, model_path: str, confidence: float, imgsz: int) -> None:
		self.detector = build_detector(model_path, confidence, imgsz)

	def set_mode(self, mode: str) -> None:
		self.mode = mode

	def set_counting_method(self, method: str) -> None:
		allowed = {"linezone", "centroid", "bottom", "bbox"}
		self._count_method = method if method in allowed else "linezone"
		for lc in self._line_counters:
			lc.set_method(self._count_method)

	def get_counting_method(self) -> str:
		return self._count_method

	def set_line(self, line: Tuple[int, int, int, int]) -> None:
		# Backward compatibility: set single line
		if not self._line_counters:
			self._line_counters = [LineCounter(line, method=self._count_method)]
		else:
			# Replace first; preserve counts internally
			self._line_counters[0].set_line(line)
			# Remove any extra lines
			if len(self._line_counters) > 1:
				self._line_counters = self._line_counters[:1]

	def set_lines(self, lines: list[Tuple[int, int, int, int]]) -> None:
		# Replace all lines, preserving counts when possible by position
		new_counters: list[LineCounter] = []
		for i, ln in enumerate(lines):
			if i < len(self._line_counters):
				self._line_counters[i].set_line(ln)
				new_counters.append(self._line_counters[i])
			else:
				new_counters.append(LineCounter(ln, method=self._count_method))
		self._line_counters = new_counters

	def add_line(self, line: Tuple[int, int, int, int]) -> None:
		self._line_counters.append(LineCounter(line))

	def remove_line_at(self, index: int) -> None:
		if 0 <= index < len(self._line_counters):
			self._line_counters.pop(index)

	def set_line_band(self, band_px: int) -> None:
		# band in pixels around the counting line; 0 to disable
		self._roi_band_px = max(0, int(band_px))

	def reset_counts(self) -> None:
		for lc in self._line_counters:
			lc.reset()

	def _approximate_head_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
		if boxes_xyxy.size == 0:
			return boxes_xyxy
		heads = boxes_xyxy.copy()
		# Take top 35% of the person box as head region, and narrow width to 60% centered
		w = heads[:, 2] - heads[:, 0]
		h = heads[:, 3] - heads[:, 1]
		heads[:, 3] = heads[:, 1] + 0.35 * h
		cx = heads[:, 0] + 0.5 * w
		new_w = 0.6 * w
		heads[:, 0] = cx - 0.5 * new_w
		heads[:, 2] = cx + 0.5 * new_w
		return heads

	def _roi_bounds(self) -> Tuple[int, int, int, int] | None:
		# Compute bounding rectangle enclosing all lines
		if not self._line_counters:
			return None
		xs: list[int] = []
		ys: list[int] = []
		for lc in self._line_counters:
			x1, y1, x2, y2 = lc._coords
			xs.extend([int(x1), int(x2)])
			ys.extend([int(y1), int(y2)])
		if not xs or not ys:
			return None
		return (min(xs), min(ys), max(xs), max(ys))

	def process(self, frame):
		assert self.detector is not None
		# Optional ROI cropping around the counting lines (union rectangle)
		roi_frame = frame
		roi_origin = (0, 0)
		if self._roi_band_px > 0:
			bounds = self._roi_bounds()
			if bounds is not None:
				bx1, by1, bx2, by2 = bounds
				h, w = frame.shape[:2]
				xmin = max(0, min(bx1, bx2) - self._roi_band_px)
				xmax = min(w, max(bx1, bx2) + self._roi_band_px)
				ymin = max(0, min(by1, by2) - self._roi_band_px)
				ymax = min(h, max(by1, by2) + self._roi_band_px)
				# ensure valid non-empty rectangle
				if xmax <= xmin:
					xmax = min(w, xmin + 1)
				if ymax <= ymin:
					ymax = min(h, ymin + 1)
				roi_frame = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
				roi_origin = (int(xmin), int(ymin))

		# Timing: detector inference
		_t0 = cv2.getTickCount()
		res = self.detector.infer(roi_frame)
		_t1 = cv2.getTickCount()
		boxes_xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4), dtype=np.float32)
		scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,), dtype=np.float32)
		classes = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,), dtype=int)

		# Filter class 0 for both models (person or face)
		mask = classes == 0
		boxes_xyxy = boxes_xyxy[mask]
		scores = scores[mask]
		classes = classes[mask]

		# If in head mode but running a body model, approximate head region
		if self.mode == "head":
			try:
				names = getattr(self.detector.model, "names", None)
				is_body = False
				if isinstance(names, (list, tuple)) and len(names) > 0:
					is_body = str(names[0]).lower() == "person"
				elif isinstance(names, dict) and 0 in names:
					is_body = str(names[0]).lower() == "person"
				if is_body:
					boxes_xyxy = self._approximate_head_boxes(boxes_xyxy)
			except Exception:
				pass

		# If cropped, map boxes back to full-frame coordinates
		if roi_origin != (0, 0) and boxes_xyxy.size > 0:
			ox, oy = roi_origin
			boxes_xyxy = boxes_xyxy.copy()
			boxes_xyxy[:, [0, 2]] += float(ox)
			boxes_xyxy[:, [1, 3]] += float(oy)

		detections = sv.Detections(xyxy=boxes_xyxy, confidence=scores, class_id=classes)
		# Timing: tracking
		_t2 = cv2.getTickCount()
		detections = self.tracker.update_with_detections(detections)
		_t3 = cv2.getTickCount()

		num = len(detections)
		tids = detections.tracker_id if detections.tracker_id is not None else [None] * num
		confs = detections.confidence if detections.confidence is not None else [0.0] * num
		labels: list[str] = []
		for tid, conf in zip(tids, confs):
			c = float(conf)
			labels.append(f"id {int(tid)} {c:.2f}" if tid is not None else f"{c:.2f}")

		# Update and draw trails first so boxes render on top
		self.viz.update_trails(detections)
		frame = self.viz.draw_trails(frame)
		# ROI band visualization (optional UI toggle handled upstream)
		try:
			bounds = self._roi_bounds()
			if bounds is not None:
				frame = self.viz.draw_roi_band(frame, bounds, self._roi_band_px)
		except Exception:
			pass
		frame = self.viz.draw_detections(frame, detections, labels)
		for lc in self._line_counters:
			lc.update(detections)
		for lc in self._line_counters:
			frame = lc.draw(frame)
		# Telemetry overlay
		freq = cv2.getTickFrequency()
		infer_ms = 1000.0 * float((_t1 - _t0) / freq)
		track_ms = 1000.0 * float((_t3 - _t2) / freq)
		overlay = frame.copy()
		cv2.rectangle(overlay, (10, 44), (10 + 300, 90), (0, 0, 0), -1)
		cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
		cv2.putText(frame, f"Infer: {infer_ms:.1f}ms  Track: {track_ms:.1f}ms", (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
		# Aggregate counts across all lines
		total_in = sum(lc.counts.incoming for lc in self._line_counters)
		total_out = sum(lc.counts.outgoing for lc in self._line_counters)
		frame = self.viz.draw_counts(frame, total_in, total_out)
		return frame

	@property
	def counts(self) -> Tuple[int, int]:
		total_in = sum(lc.counts.incoming for lc in self._line_counters)
		total_out = sum(lc.counts.outgoing for lc in self._line_counters)
		return total_in, total_out
