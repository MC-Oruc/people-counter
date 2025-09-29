from __future__ import annotations

from typing import Optional
import numpy as np
import cv2

try:
	from ultralytics import YOLO  # type: ignore
except Exception:
	YOLO = None  # type: ignore

try:
	import torch  # type: ignore
except Exception:
	torch = None  # type: ignore


class _Boxes:
	def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
		self.xyxy = self._to_tensor_like(xyxy)
		self.conf = self._to_tensor_like(conf)
		self.cls = self._to_tensor_like(cls)

	@staticmethod
	def _to_tensor_like(arr: np.ndarray):
		class _T:
			def __init__(self, a: np.ndarray) -> None:
				self._a = a
			def cpu(self):
				return self
			def numpy(self):
				return self._a
		return _T(arr)


class _Result:
	def __init__(self, boxes: _Boxes) -> None:
		self.boxes = boxes


class YoloDetector:
	def __init__(self, model_path: str, confidence: float = 0.3, imgsz: int = 640, device: str = "cpu") -> None:
		if YOLO is None:
			raise RuntimeError("Ultralytics yuklenemedi. Lütfen kurulum betigini calistirin.")
		self.model = YOLO(model_path)
		self.confidence = confidence
		self.imgsz = imgsz
		# centralized device helpers
		from .utils.device import resolve_device, default_half_for
		self.device = resolve_device(device)
		self.use_half: bool = default_half_for(self.device)
		self._apply_device()

	def is_gpu_capable(self) -> bool:
		return True

	def _resolve_device(self, device: str) -> str:
		from .utils.device import resolve_device
		return resolve_device(device)

	def _apply_device(self) -> None:
		try:
			self.model.to(self.device)
		except Exception:
			pass

	def set_device(self, device: str) -> None:
		self.device = self._resolve_device(device)
		self._apply_device()
		# Disable half automatically if not CUDA
		from .utils.device import default_half_for
		self.use_half = default_half_for(self.device)

	def get_device(self) -> str:
		return self.device

	def set_half(self, use_half: bool) -> None:
		# half only meaningful on CUDA
		self.use_half = bool(use_half) and str(self.device).startswith('cuda')

	def get_half(self) -> bool:
		return bool(self.use_half)

	def set_confidence(self, confidence: float) -> None:
		self.confidence = float(confidence)

	def infer(self, frame_bgr: np.ndarray):
		results = self.model(
			source=frame_bgr,
			conf=self.confidence,
			device=self.device,
			imgsz=self.imgsz,
			classes=[0],
			half=(self.use_half and str(self.device).startswith('cuda')),
			verbose=False,
		)
		return results[0]


class YuNetFaceDetector:
	def __init__(self, onnx_path: str, score_threshold: float = 0.6) -> None:
		self.model_path = onnx_path
		self.score_threshold = float(score_threshold)
		self.net = cv2.FaceDetectorYN_create(self.model_path, "", (320, 320), self.score_threshold, 0.3, 5000)
		self._last_size: Optional[tuple[int, int]] = None

	def is_gpu_capable(self) -> bool:
		return False

	def set_device(self, device: str) -> None:
		# Not applicable for YuNet (CPU via OpenCV backend)
		pass

	def get_device(self) -> str:
		return "cpu"

	def set_confidence(self, confidence: float) -> None:
		self.score_threshold = float(confidence)
		try:
			self.net.setScoreThreshold(self.score_threshold)
		except Exception:
			pass

	def infer(self, frame_bgr: np.ndarray):
		h, w = frame_bgr.shape[:2]
		if self._last_size != (w, h):
			self.net.setInputSize((w, h))
			self._last_size = (w, h)
		ok, faces = self.net.detect(frame_bgr)
		if not ok or faces is None or len(faces) == 0:
			xyxy = np.zeros((0, 4), dtype=np.float32)
			conf = np.zeros((0,), dtype=np.float32)
			cls = np.zeros((0,), dtype=np.int32)
			return _Result(_Boxes(xyxy, conf, cls))
		faces = faces.reshape(-1, 15)
		x, y, ww, hh, s = faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 3], faces[:, 4]
		xyxy = np.stack([x, y, x + ww, y + hh], axis=1).astype(np.float32)
		conf = s.astype(np.float32)
		cls = np.zeros((xyxy.shape[0],), dtype=np.int32)
		return _Result(_Boxes(xyxy, conf, cls))


def build_detector(model_path: str, confidence: float, imgsz: int, device: str = "cpu"):
	mp = model_path.lower()
	if mp.endswith(".onnx") or "yunet" in mp:
		return YuNetFaceDetector(model_path, score_threshold=confidence)
	return YoloDetector(model_path, confidence, imgsz, device)


# Backwards compatibility: deprecated alias (kept for external imports)
PersonDetector = YoloDetector
