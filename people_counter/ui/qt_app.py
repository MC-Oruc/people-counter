from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from collections import deque
from PySide6 import QtCore, QtGui, QtWidgets
import threading

from ..core.engine import PeopleEngine
from ..utils.line_io import (
	load_line,
	save_line,
	load_lines,
	save_lines,
	load_lines_for_source,
	load_line_for_source,
	save_lines_for_source,
	save_line_for_source,
)
from ..utils.capture import open_capture, is_rtsp_url
from ..config import ensure_default_configs, DEFAULT_APP_CFG
import yaml


@dataclass
class AppState:
	source: str | int = "0"
	model_path: str = "yolov8n.pt"
	confidence: float = 0.3
	imgsz: int = 640
	running: bool = False
	recording: bool = False
	line: Tuple[int, int, int, int] = (100, 200, 500, 200)
	lines: list[Tuple[int, int, int, int]] = field(default_factory=list)
	selected_line_idx: int = 0
	counts_in: int = 0
	counts_out: int = 0
	mode: str = "body"  # or "head"
	video_speed_mult: float = 1.0  # 1.0x by default
	count_method: str = "linezone"  # 'linezone'|'centroid'|'bottom'|'bbox'


def bgr_to_qimage(frame: np.ndarray) -> QtGui.QImage:
	if frame.ndim == 2:
		h, w = frame.shape
		fmt = QtGui.QImage.Format_Grayscale8
		return QtGui.QImage(frame.data, w, h, w, fmt).copy()
	elif frame.ndim == 3:
		h, w, ch = frame.shape
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		bytes_per_line = ch * w
		return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
	raise ValueError("Unsupported image shape")


class VideoWidget(QtWidgets.QLabel):
	lineChanged = QtCore.Signal(tuple)
	linesChanged = QtCore.Signal(list)

	def __init__(self) -> None:
		super().__init__()
		self.setScaledContents(False)
		self.setMinimumSize(640, 360)
		self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
		self.setStyleSheet("background-color: #000000;")
		self._frame_size: Optional[Tuple[int, int]] = None  # (w,h)
		self._lines: list[Tuple[int, int, int, int]] = [(100, 200, 500, 200)]
		self._sel_idx: int = 0
		self._drag: Optional[str] = None
		self._grab_r = 12
		self._edit_enabled: bool = True

	def enable_edit(self, enabled: bool) -> None:
		self._edit_enabled = enabled
		self.update()

	def set_frame_size(self, w: int, h: int) -> None:
		self._frame_size = (w, h)

	def set_line(self, line: Tuple[int, int, int, int]) -> None:
		# Backward compatibility: set first line
		if not self._lines:
			self._lines = [line]
			self._sel_idx = 0
		else:
			self._lines[0] = line
			self._sel_idx = min(self._sel_idx, len(self._lines) - 1)
		self.update()
		self.lineChanged.emit(self.get_line())
		self.linesChanged.emit(self.get_lines())

	def get_line(self) -> Tuple[int, int, int, int]:
		if not self._lines:
			return (100, 200, 500, 200)
		idx = min(max(0, self._sel_idx), len(self._lines) - 1)
		return self._lines[idx]

	def set_lines(self, lines: list[Tuple[int, int, int, int]]) -> None:
		self._lines = [
			(int(l[0]), int(l[1]), int(l[2]), int(l[3])) for l in (lines or [])
		]
		self._sel_idx = 0 if self._lines else -1
		self.update()
		if self._lines:
			self.lineChanged.emit(self.get_line())
		self.linesChanged.emit(self.get_lines())

	def get_lines(self) -> list[Tuple[int, int, int, int]]:
		return list(self._lines)

	def set_selected_index(self, idx: int) -> None:
		if not self._lines:
			self._sel_idx = -1
			self.update()
			return
		self._sel_idx = max(0, min(int(idx), len(self._lines) - 1))
		self.update()
		self.lineChanged.emit(self.get_line())

	def add_line(self, line: Tuple[int, int, int, int]) -> None:
		self._lines.append((int(line[0]), int(line[1]), int(line[2]), int(line[3])))
		self._sel_idx = len(self._lines) - 1
		self.update()
		self.lineChanged.emit(self.get_line())
		self.linesChanged.emit(self.get_lines())

	def remove_line_at(self, idx: int) -> None:
		if 0 <= idx < len(self._lines):
			self._lines.pop(idx)
			if not self._lines:
				self._sel_idx = -1
			else:
				self._sel_idx = max(0, min(self._sel_idx, len(self._lines) - 1))
			self.update()
			self.linesChanged.emit(self.get_lines())

	def display_qimage(self, img: QtGui.QImage) -> None:
		# Always scale with aspect ratio preserved and centered
		pix = QtGui.QPixmap.fromImage(img)
		scaled = pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
		self.setPixmap(scaled)
		self.update()

	def paintEvent(self, event: QtGui.QPaintEvent) -> None:
		super().paintEvent(event)
		if self.pixmap() is None:
			return
		p = QtGui.QPainter(self)
		p.setRenderHint(QtGui.QPainter.Antialiasing)
		# draw line and handles in label coordinates
		if self._frame_size is None:
			p.end(); return
		fx, fy = self._frame_size
		if fx <= 0 or fy <= 0:
			p.end(); return
		# compute scale used in pixmap.scaled by QLabel auto-scale above
		sx = self.pixmap().width() / fx
		sy = self.pixmap().height() / fy
		# find top-left offset for centered pixmap
		off_x = (self.width() - self.pixmap().width()) // 2
		off_y = (self.height() - self.pixmap().height()) // 2
		# draw all lines
		for i, (lx1, ly1, lx2, ly2) in enumerate(self._lines):
			x1, y1 = int(lx1 * sx) + off_x, int(ly1 * sy) + off_y
			x2, y2 = int(lx2 * sx) + off_x, int(ly2 * sy) + off_y
			color = QtGui.QColor(255, 255, 0) if i != self._sel_idx else QtGui.QColor(255, 215, 0)
			pen = QtGui.QPen(color, 2)
			p.setPen(pen)
			p.drawLine(x1, y1, x2, y2)
			if self._edit_enabled and i == self._sel_idx:
				p.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 2))
				p.drawEllipse(QtCore.QPoint(x1, y1), self._grab_r, self._grab_r)
				p.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 2))
				p.drawEllipse(QtCore.QPoint(x2, y2), self._grab_r, self._grab_r)
		p.end()

	def _hit(self, x: int, y: int, cx: int, cy: int) -> bool:
		dx, dy = x - cx, y - cy
		return dx * dx + dy * dy <= self._grab_r * self._grab_r

	def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
		if not self._edit_enabled or self._frame_size is None or self.pixmap() is None:
			return
		fx, fy = self._frame_size
		sx = self.pixmap().width() / fx
		sy = self.pixmap().height() / fy
		off_x = (self.width() - self.pixmap().width()) // 2
		off_y = (self.height() - self.pixmap().height()) // 2
		if not self._lines or self._sel_idx < 0 or self._sel_idx >= len(self._lines):
			return
		lx1, ly1, lx2, ly2 = self._lines[self._sel_idx]
		x1, y1 = int(lx1 * sx) + off_x, int(ly1 * sy) + off_y
		x2, y2 = int(lx2 * sx) + off_x, int(ly2 * sy) + off_y
		pos = event.position().toPoint()
		if self._hit(pos.x(), pos.y(), x1, y1):
			self._drag = 'p1'
		elif self._hit(pos.x(), pos.y(), x2, y2):
			self._drag = 'p2'

	def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
		if not self._edit_enabled or self._drag is None or self._frame_size is None or self.pixmap() is None:
			return
		fx, fy = self._frame_size
		sx = self.pixmap().width() / fx
		sy = self.pixmap().height() / fy
		off_x = (self.width() - self.pixmap().width()) // 2
		off_y = (self.height() - self.pixmap().height()) // 2
		pos = event.position().toPoint()
		lx1, ly1, lx2, ly2 = self._lines[self._sel_idx]
		if self._drag == 'p1':
			lx1 = int((pos.x() - off_x) / max(1e-6, sx))
			ly1 = int((pos.y() - off_y) / max(1e-6, sy))
		else:
			lx2 = int((pos.x() - off_x) / max(1e-6, sx))
			ly2 = int((pos.y() - off_y) / max(1e-6, sy))
		self._lines[self._sel_idx] = (
			max(0, lx1), max(0, ly1),
			max(0, lx2), max(0, ly2)
		)
		self.lineChanged.emit(self.get_line())
		self.linesChanged.emit(self.get_lines())
		self.update()

	def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
		self._drag = None


class LoadingOverlay(QtWidgets.QWidget):
	"""Semi-transparent overlay with spinner and status text for connection attempts."""

	def __init__(self, parent: QtWidgets.QWidget) -> None:
		super().__init__(parent)
		self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
		self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
		self._angle: int = 0
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self._on_tick)
		self._text: str = ""
		self.hide()

	def set_text(self, text: str) -> None:
		self._text = text
		self.update()

	def start(self) -> None:
		if not self.isVisible():
			self.show()
		self._timer.start(50)
		self.update()

	def stop(self) -> None:
		self._timer.stop()
		self.hide()

	def _on_tick(self) -> None:
		self._angle = (self._angle + 12) % 360
		self.update()

	def paintEvent(self, event: QtGui.QPaintEvent) -> None:
		if not self.isVisible():
			return
		p = QtGui.QPainter(self)
		p.setRenderHint(QtGui.QPainter.Antialiasing)
		# dim background
		p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 140))
		# spinner (smaller: half radius -> ~quarter area)
		cx, cy = self.width() // 2, self.height() // 2
		base_r = max(16, min(self.width(), self.height()) // 10)
		r = max(12, base_r // 2)
		pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 2)
		pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
		p.setPen(pen)
		# draw arc 270 degrees, rotated by angle
		start_angle = int(self._angle * 16)
		span_angle = int(270 * 16)
		rect = QtCore.QRect(cx - r, cy - r + 6, 2 * r, 2 * r)
		p.drawArc(rect, start_angle, span_angle)
		# text
		if self._text:
			p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
			font = p.font()
			font.setPointSize(11)
			p.setFont(font)
			metrics = QtGui.QFontMetrics(font)
			text_w = metrics.horizontalAdvance(self._text)
			# draw text ABOVE the spinner
			p.drawText(cx - text_w // 2, cy - r - 12, self._text)
		p.end()


class MainWindow(QtWidgets.QMainWindow):
	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("Kisi Sayaci GUI")
		self.resize(1100, 720)

		self.state = AppState()
		self.engine = PeopleEngine()

		self.cap: Optional[cv2.VideoCapture] = None
		self.writer: Optional[cv2.VideoWriter] = None
		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self.on_tick)
		self.fps = 25.0
		self.base_fps = 25.0
		self.frame_size: Optional[Tuple[int, int]] = None

		self._build_ui()
		self._connect_signals()

	def _init_count_method_combo(self) -> None:
		# Order: descriptive labels mapping to internal keys
		self.cmb_count_method.addItem("Varsayilan (LineZone)", userData="linezone")
		self.cmb_count_method.addItem("Merkez Nokta", userData="centroid")
		self.cmb_count_method.addItem("Alt Kenar (ayaklar)", userData="bottom")
		self.cmb_count_method.addItem("BBox Yol Kesişimi", userData="bbox")
		# Set default
		self.cmb_count_method.setCurrentIndex(0)

	def _build_ui(self) -> None:
		central = QtWidgets.QWidget()
		self.setCentralWidget(central)

		self.video = VideoWidget()
		self.slider_seek = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
		self.slider_seek.setEnabled(False)
		self.slider_seek.setRange(0, 0)
		self.slider_seek.setSingleStep(1)
		self.slider_seek.setPageStep(1)

		# Controls
		self.btn_open = QtWidgets.QPushButton("Kaynak Ac/Baslat")
		self.btn_toggle = QtWidgets.QPushButton("Taramayi Baslat")
		self.btn_record = QtWidgets.QPushButton("Kaydi Baslat")
		self.btn_reset = QtWidgets.QPushButton("Sayaci Sifirla")
		self.btn_save = QtWidgets.QPushButton("Cizgileri Kaydet")
		self.chk_edit = QtWidgets.QCheckBox("Cizgiyi Duzenle")
		self.chk_edit.setChecked(True)
		self.chk_show_trails = QtWidgets.QCheckBox("Izleri Goster")
		self.chk_show_trails.setChecked(True)
		self.chk_rtsp_quality = QtWidgets.QCheckBox("RTSP: Kaliteyi Tercih Et")
		self.chk_rtsp_quality.setToolTip("RTSP icin daha buyuk tampon ve zaman asimlari (daha az artifakt, daha yuksek gecikme)")

		# Source categories and per-category controls
		self.cmb_source_category = QtWidgets.QComboBox()
		self.cmb_source_category.addItems(["Kamera", "Video", "RTSP"])
		# Camera panel
		self.cmb_camera_index = QtWidgets.QComboBox()
		self.btn_refresh_cameras = QtWidgets.QPushButton("Yenile")
		_cam_row = QtWidgets.QHBoxLayout()
		_cam_row.addWidget(QtWidgets.QLabel("Kamera No:"))
		_cam_row.addWidget(self.cmb_camera_index, 1)
		_cam_row.addWidget(self.btn_refresh_cameras)
		camera_panel = QtWidgets.QWidget()
		camera_panel.setLayout(_cam_row)
		# Video panel
		self.cmb_video_list = QtWidgets.QComboBox()
		self.cmb_video_list.setEditable(True)
		self.cmb_video_list.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.InsertAtTop)
		self.cmb_video_list.setMinimumWidth(220)
		self.btn_browse_video = QtWidgets.QPushButton("Gozat...")
		_vid_row = QtWidgets.QHBoxLayout()
		_vid_row.addWidget(self.cmb_video_list, 1)
		_vid_row.addWidget(self.btn_browse_video)
		video_panel = QtWidgets.QWidget()
		video_panel.setLayout(_vid_row)
		# RTSP panel
		self.edit_rtsp_url = QtWidgets.QLineEdit()
		self.edit_rtsp_url.setPlaceholderText("rtsp://kullanici:sifre@host:554/stream")
		_rtsp_col = QtWidgets.QVBoxLayout()
		_rtsp_col.addWidget(self.edit_rtsp_url)
		_rtsp_col.addWidget(self.chk_rtsp_quality)
		rtsp_panel = QtWidgets.QWidget()
		rtsp_panel.setLayout(_rtsp_col)
		# Stacked widget
		self.source_stack = QtWidgets.QStackedWidget()
		self.source_stack.addWidget(camera_panel)
		self.source_stack.addWidget(video_panel)
		self.source_stack.addWidget(rtsp_panel)

		# Video controls (enabled only for file sources)
		self.lbl_speed = QtWidgets.QLabel("Hiz: 1.00x")
		self.slider_speed = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
		self.slider_speed.setRange(10, 1700)  # 0.10x .. 17.00x
		self.slider_speed.setValue(int(self.state.video_speed_mult * 100))
		self.slider_speed.setSingleStep(5)
		self.model_edit = QtWidgets.QLineEdit("models/yolov8n.pt")
		self.cmb_model = QtWidgets.QComboBox()
		self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
		self.conf_slider.setRange(5, 90)
		self.conf_slider.setValue(int(self.state.confidence * 100))
		self.lbl_conf = QtWidgets.QLabel(f"Guven: {self.state.confidence:.2f}")

		self.cmb_mode = QtWidgets.QComboBox()
		self.cmb_mode.addItems(["body", "head"])  # Body / Head
		# Counting method selector
		self.cmb_count_method = QtWidgets.QComboBox()
		self._init_count_method_combo()
		self.cmb_count_method.setToolTip("Sayma mantigi:\n- Varsayilan (LineZone): Supervision LineZone (merkez tabanli)\n- Merkez Nokta: BBox merkezi cizgiyi karsiladiginda\n- Alt Kenar: BBox'in alt kenari (ayaklar) cizgiyi gectiginde\n- BBox Yol Kesisimi: Tum kutu once bir tarafta, sonra diger tarafta oldugunda")
		self.cmb_device = QtWidgets.QComboBox()
		self.cmb_device.addItems(["CPU", "GPU"])  # device selector
		self.cmb_device_variant = QtWidgets.QComboBox()  # e.g., cuda:0, cuda:1, cpu
		self.cmb_backend = QtWidgets.QComboBox()  # available GPU modules/backends
		self.chk_half = QtWidgets.QCheckBox("CUDA FP16 (half)")
		self.chk_half.setToolTip("CUDA'da FP16 ile inferans (hizli, bazi cihazlarda daha az bellek)")

		# ROI band controls
		self.spin_roi_band = QtWidgets.QSpinBox()
		self.spin_roi_band.setRange(0, 512)
		self.spin_roi_band.setValue(0)
		self.chk_roi_show = QtWidgets.QCheckBox("ROI Bandini Goster")
		self.chk_roi_show.setChecked(True)

		self.lbl_counts = QtWidgets.QLabel("Giris: 0 | Cikis: 0")

		self.form = QtWidgets.QFormLayout()
		self.form.addRow("Kaynak Tipi:", self.cmb_source_category)
		self.form.addRow("Kaynak:", self.source_stack)
		_vrow = QtWidgets.QHBoxLayout()
		_vrow.addWidget(self.lbl_speed)
		_vrow.addWidget(self.slider_speed)
		_vrow_w = QtWidgets.QWidget()
		_vrow_w.setLayout(_vrow)
		self._lbl_video_controls = QtWidgets.QLabel("Video Kontrolleri:")
		self.form.addRow(self._lbl_video_controls, _vrow_w)
		_model_row = QtWidgets.QHBoxLayout()
		_model_row.addWidget(self.cmb_model)
		_model_row.addWidget(self.model_edit)
		_model_row_w = QtWidgets.QWidget()
		_model_row_w.setLayout(_model_row)
		self.form.addRow("Model:", _model_row_w)
		self.form.addRow(self.lbl_conf, self.conf_slider)
		self.form.addRow("Hedef:", self.cmb_mode)
		self.form.addRow("Sayma Yöntemi:", self.cmb_count_method)
		self.form.addRow("Aygit:", self.cmb_device)
		self._lbl_dev_variant = QtWidgets.QLabel("Aygit Secenek:")
		self.form.addRow(self._lbl_dev_variant, self.cmb_device_variant)
		self._lbl_backend = QtWidgets.QLabel("GPU Modulleri:")
		self.form.addRow(self._lbl_backend, self.cmb_backend)
		self._lbl_half = QtWidgets.QLabel("CUDA Secenek:")
		self.form.addRow(self._lbl_half, self.chk_half)
		_roi_row = QtWidgets.QHBoxLayout()
		_roi_row.addWidget(self.spin_roi_band)
		_roi_row.addWidget(self.chk_roi_show)
		_roi_row_w = QtWidgets.QWidget()
		_roi_row_w.setLayout(_roi_row)
		self.form.addRow("ROI Band (px):", _roi_row_w)

		# Lines management group
		self.list_lines = QtWidgets.QListWidget()
		self.list_lines.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
		self.btn_add_line = QtWidgets.QPushButton("Cizgi Ekle")
		self.btn_remove_line = QtWidgets.QPushButton("Seciliyi Sil")
		_lines_btn_row = QtWidgets.QHBoxLayout()
		_lines_btn_row.addWidget(self.btn_add_line)
		_lines_btn_row.addWidget(self.btn_remove_line)
		_lines_group_layout = QtWidgets.QVBoxLayout()
		_lines_group_layout.addWidget(self.list_lines)
		_lines_group_layout.addLayout(_lines_btn_row)
		_lines_group = QtWidgets.QGroupBox("Cizgiler")
		_lines_group.setLayout(_lines_group_layout)

		buttons_layout = QtWidgets.QGridLayout()
		buttons_layout.addWidget(self.btn_open, 0, 0)
		buttons_layout.addWidget(self.btn_toggle, 0, 1)
		buttons_layout.addWidget(self.btn_record, 0, 2)
		buttons_layout.addWidget(self.btn_reset, 0, 3)
		buttons_layout.addWidget(self.btn_save, 1, 1)
		buttons_layout.addWidget(self.chk_edit, 1, 2)
		buttons_layout.addWidget(self.chk_show_trails, 1, 3)
		# RTSP kalite secenegi RTSP panelinde gosteriliyor
		# Stream metrics label
		self.lbl_stream = QtWidgets.QLabel("Akis: - | Gecikme: - | FPS: -")
		buttons_layout.addWidget(self.lbl_counts, 2, 1, 1, 3)
		buttons_layout.addWidget(self.lbl_stream, 3, 1, 1, 3)

		# Build right panel widget
		right_panel = QtWidgets.QWidget()
		right = QtWidgets.QVBoxLayout(right_panel)
		right.addLayout(self.form)
		right.addWidget(_lines_group)
		right.addLayout(buttons_layout)
		right.addStretch(1)

		# Build left panel widget
		left_panel = QtWidgets.QWidget()
		left_col = QtWidgets.QVBoxLayout(left_panel)
		left_col.addWidget(self.video)
		left_col.addWidget(self.slider_seek)

		# Loading overlay on top of the video
		self.overlay = LoadingOverlay(self.video)
		self.overlay.setGeometry(self.video.rect())
		# Keep overlay sized with video widget
		self.video.installEventFilter(self)

		# Resizable splitter
		splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
		splitter.addWidget(left_panel)
		splitter.addWidget(right_panel)
		splitter.setStretchFactor(0, 3)
		splitter.setStretchFactor(1, 2)

		root = QtWidgets.QVBoxLayout(central)
		root.addWidget(splitter)

		# Removed toolbar per user preference; actions remain accessible via buttons

		# Status bar: move counts and stream info here
		self.setStatusBar(QtWidgets.QStatusBar())
		self.statusBar().addPermanentWidget(self.lbl_counts)
		self.statusBar().addPermanentWidget(self.lbl_stream)

	def _connect_signals(self) -> None:
		self.btn_open.clicked.connect(self.on_open)
		self.btn_toggle.clicked.connect(self.on_toggle)
		self.btn_record.clicked.connect(self.on_record)
		self.btn_reset.clicked.connect(self.on_reset)
		self.btn_save.clicked.connect(self.on_save_line)
		self.btn_add_line.clicked.connect(self.on_add_line)
		self.btn_remove_line.clicked.connect(self.on_remove_line)
		self.chk_edit.toggled.connect(self.video.enable_edit)
		self.chk_show_trails.toggled.connect(self.on_show_trails_toggled)
		self.conf_slider.valueChanged.connect(self.on_conf_changed)
		self.video.lineChanged.connect(self.on_line_changed)
		self.video.linesChanged.connect(self.on_lines_changed)
		self.cmb_mode.currentTextChanged.connect(self.on_mode_changed)
		self.cmb_device.currentTextChanged.connect(self.on_device_changed)
		self.cmb_device_variant.currentTextChanged.connect(self.on_device_variant_changed)
		self.cmb_model.currentTextChanged.connect(self.on_model_selected)
		self.chk_half.toggled.connect(self.on_half_toggled)
		self.spin_roi_band.valueChanged.connect(self.on_roi_band_changed)
		self.chk_roi_show.toggled.connect(self.on_roi_show_toggled)
		self.cmb_count_method.currentIndexChanged.connect(self.on_count_method_changed)
		self.cmb_source_category.currentIndexChanged.connect(self.on_source_category_changed)
		self.btn_browse_video.clicked.connect(self.on_browse_video)
		self.btn_refresh_cameras.clicked.connect(self.on_refresh_cameras)
		self.slider_speed.valueChanged.connect(self.on_speed_changed)
		self.slider_seek.sliderPressed.connect(self.on_seek_pressed)
		self.slider_seek.sliderReleased.connect(self.on_seek_released)
		self.slider_seek.valueChanged.connect(self.on_seek_changed)
		self.list_lines.currentRowChanged.connect(self.on_selected_line_changed)

		# Populate models list
		self._populate_models_list()
		# Populate source lists (cameras/videos)
		self._populate_sources_list()
		# Auto load line at startup
		self._auto_load_line()
		# Disable video controls initially
		self._update_video_controls_enabled(False)
		# Seek disabled initially
		self.slider_seek.setEnabled(False)

		# Internal playback state
		self.is_video: bool = False
		self.last_frame: Optional[np.ndarray] = None
		self.total_frames: int = -1
		self.cur_frame_idx: int = 0
		self._is_scrubbing: bool = False

		# Initialize lines UI state
		self._sync_lines_list_widget()

		# Load persisted UI settings (from config/app.yaml) after controls are ready
		try:
			self._load_ui_config()
		except Exception:
			pass
		# Reflect initial category state for controls
		try:
			self.on_source_category_changed(self.cmb_source_category.currentIndex())
		except Exception:
			pass
		# Reflect device-dependent visibility
		try:
			self._update_device_controls_visibility()
		except Exception:
			pass

		# Background worker state
		self._proc_queue: deque[np.ndarray] = deque(maxlen=2)
		self._proc_thread: Optional[threading.Thread] = None
		self._proc_stop = threading.Event()
		# Async writer state
		self._writer_queue: deque[np.ndarray] = deque(maxlen=16)
		self._writer_thread: Optional[threading.Thread] = None
		self._writer_stop = threading.Event()
		# Background capture state (for RTSP/streams to avoid UI blocking)
		self._cap_queue: deque[tuple[np.ndarray, float]] = deque(maxlen=16)
		self._cap_thread: Optional[threading.Thread] = None
		self._cap_stop = threading.Event()
		self._cap_reconnect_backoff_s: float = 2.0
		# Stream metrics
		self._cap_prev_ts: Optional[float] = None
		self._cap_fps_ewma: float = 0.0
		self._cap_jitter_ms_ewma: float = 0.0
		self._cap_drop_count: int = 0
		self._ui_frames_accum: int = 0
		self._ui_last_rate_time: float = time.time()
		self._ui_fps_ewma: float = 0.0
		self._rtsp_latency_ms_ewma: float = 0.0
		self._last_stream_label_update: float = 0.0
		# Connection indicator state
		self._is_connecting: bool = False
		self._connecting_reason: str = ""  # 'rtsp' | 'camera'
		self._connecting_started: float = 0.0
		self._connect_popup_shown: bool = False
		self._request_connecting_overlay: bool = False

	def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
		if obj is self.video and event.type() == QtCore.QEvent.Type.Resize:
			self.overlay.setGeometry(self.video.rect())
		return super().eventFilter(obj, event)
	def on_selected_line_changed(self, row: int) -> None:
		self.state.selected_line_idx = int(row)
		self.video.set_selected_index(int(row))

	def _update_seek_controls(self) -> None:
		enabled = self.is_video and (self.total_frames is not None) and (self.total_frames > 0)
		self.slider_seek.setEnabled(bool(enabled))
		if enabled:
			self.slider_seek.blockSignals(True)
			self.slider_seek.setRange(0, max(0, int(self.total_frames) - 1))
			cur = max(0, int(self.cur_frame_idx) - 1)
			self.slider_seek.setValue(cur)
			pg = max(1, (int(self.total_frames) // 100))
			self.slider_seek.setPageStep(pg)
			self.slider_seek.blockSignals(False)
		else:
			self.slider_seek.blockSignals(True)
			self.slider_seek.setRange(0, 0)
			self.slider_seek.setValue(0)
			self.slider_seek.blockSignals(False)
	def _scan_video_files(self, base: Path) -> list[str]:
		if not base.exists() or not base.is_dir():
			return []
		items: list[str] = []
		exts = {".mp4", ".avi", ".mkv", ".mov", ".mjpeg", ".mjpg", ".wmv", ".webm"}
		for p in sorted(base.iterdir()):
			if p.is_file() and p.suffix.lower() in exts:
				try:
					rel = p.relative_to(Path.cwd())
					items.append(str(rel))
				except Exception:
					items.append(str(p))
		return items

	def _populate_sources_list(self) -> None:
		# Populate camera indices (basic defaults; user can refresh to probe)
		try:
			self.cmb_camera_index.blockSignals(True)
			self.cmb_camera_index.clear()
			for d in ["0", "1", "2", "3"]:
				self.cmb_camera_index.addItem(d)
		finally:
			self.cmb_camera_index.blockSignals(False)
		# Populate videos from tests/
		videos: list[str] = []
		for folder in (Path.cwd() / "tests",):
			videos.extend(self._scan_video_files(folder))
		self.cmb_video_list.blockSignals(True)
		self.cmb_video_list.clear()
		for v in videos:
			self.cmb_video_list.addItem(v)
		self.cmb_video_list.blockSignals(False)
		self._update_video_controls_enabled(False)

	def _derive_category_from_source(self, src: str) -> int:
		# 0: Kamera, 1: Video, 2: RTSP
		try:
			if src.strip().lower().startswith("rtsp://"):
				return 2
			# numeric -> camera
			int(src)
			return 0
		except Exception:
			pass
		# file path -> video
		return 1

	def on_refresh_cameras(self) -> None:
		# Probe camera indices 0..9 quickly
		found: list[str] = []
		for idx in range(0, 10):
			try:
				cap = cv2.VideoCapture(idx)
				ok = cap.isOpened()
				try:
					cap.release()
				except Exception:
					pass
				if ok:
					found.append(str(idx))
			except Exception:
				pass
		if not found:
			found = ["0"]
		self.cmb_camera_index.blockSignals(True)
		self.cmb_camera_index.clear()
		for s in found:
			self.cmb_camera_index.addItem(s)
		self.cmb_camera_index.blockSignals(False)

	def on_browse_video(self) -> None:
		start_dir = str(Path.cwd())
		filters = "Video Files (*.mp4 *.avi *.mkv *.mov *.mjpeg *.mjpg *.wmv *.webm);;All Files (*)"
		path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Video Dosyasi Sec", start_dir, filters)
		if not path:
			return
		if self.cmb_video_list.findText(path) < 0:
			self.cmb_video_list.addItem(path)
		self.cmb_video_list.setCurrentText(path)
		self._update_video_controls_enabled(True)
		self._update_seek_controls()

	def on_source_category_changed(self, idx: int) -> None:
		self.source_stack.setCurrentIndex(max(0, int(idx)))
		# Enable video controls only for Video category
		self._update_video_controls_enabled(idx == 1)

	def on_browse_source(self) -> None:
		start_dir = str(Path.cwd())
		filters = "Video Files (*.mp4 *.avi *.mkv *.mov *.mjpeg *.mjpg *.wmv *.webm);;All Files (*)"
		path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Video Dosyasi Sec", start_dir, filters)
		if not path:
			return
		if self.cmb_source.findText(path) < 0:
			self.cmb_source.addItem(path)
		self.cmb_source.setCurrentText(path)
		self._update_video_controls_enabled(True)
		self._update_seek_controls()

	def _is_video_path(self, source: str | int) -> bool:
		if isinstance(source, int):
			return False
		try:
			p = Path(str(source))
			return p.exists() and p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov", ".mjpeg", ".mjpg", ".wmv", ".webm"}
		except Exception:
			return False

	def _update_video_controls_enabled(self, enabled: bool) -> None:
		# Hide/show instead of just enabling to declutter UI
		self.lbl_speed.setVisible(bool(enabled))
		self.slider_speed.setVisible(bool(enabled))
		self.slider_seek.setVisible(bool(enabled))
		# Also hide the form label row for Video controls
		try:
			self._lbl_video_controls.setVisible(bool(enabled))
		except Exception:
			pass

	def _update_device_controls_visibility(self) -> None:
		dev = self.cmb_device.currentText().lower()
		is_gpu = (dev == 'gpu')
		self.cmb_device_variant.setVisible(is_gpu)
		self.chk_half.setVisible(is_gpu)
		self.cmb_backend.setVisible(is_gpu)
		# Hide corresponding labels, too
		try:
			self._lbl_dev_variant.setVisible(is_gpu)
			self._lbl_backend.setVisible(is_gpu)
			self._lbl_half.setVisible(is_gpu)
		except Exception:
			pass

	def _force_cpu_selection(self) -> None:
		# Update UI to reflect CPU selection and set device on detector
		try:
			self.cmb_device.blockSignals(True)
			self.cmb_device_variant.blockSignals(True)
			self.cmb_device.setCurrentText("CPU")
			self._refresh_device_variants('cpu')
			idx = self.cmb_device_variant.findText('cpu')
			if idx >= 0:
				self.cmb_device_variant.setCurrentIndex(idx)
		finally:
			self.cmb_device.blockSignals(False)
			self.cmb_device_variant.blockSignals(False)
		try:
			if self.engine.detector is not None and hasattr(self.engine.detector, 'set_device'):
				self.engine.detector.set_device('cpu')
		except Exception:
			pass

	def on_speed_changed(self, val: int) -> None:
		# Slider maps 10..1700 -> 0.10x..17.00x
		self.state.video_speed_mult = max(0.1, min(17.0, val / 100.0))
		self.lbl_speed.setText(f"Hiz: {self.state.video_speed_mult:.2f}x")
		if self.is_video and self.state.running:
			play_fps = max(1e-3, self.base_fps * self.state.video_speed_mult)
			interval = int(max(1, round(1000.0 / play_fps)))
			self.timer.setInterval(interval)

	def on_roi_band_changed(self, val: int) -> None:
		try:
			self.engine.set_line_band(int(val))
		except Exception:
			pass

	def on_roi_show_toggled(self, checked: bool) -> None:
		# draw a visible guide on top of video widget by forcing repaint; actual ROI drawing is done in engine overlay if desired
		self.video.update()

		# removed: unused on_prev_frame/on_next_frame step controls

	def _display_frame(self, frame: np.ndarray, processing: bool) -> None:
		# Always show the raw frame; overlays are drawn in VideoWidget.paintEvent
		self.video.set_frame_size(frame.shape[1], frame.shape[0])
		qimg = bgr_to_qimage(frame)
		self.video.display_qimage(qimg)
		# enqueue to worker if running
		if processing:
			self._enqueue_for_processing(frame)

	def on_seek_pressed(self) -> None:
		if not self.is_video:
			return
		self._is_scrubbing = True
		if self.state.running:
			self.state.running = False
			self.btn_toggle.setText("Taramayi Baslat")

	def on_seek_released(self) -> None:
		if not self.is_video:
			return
		self._is_scrubbing = False
		self._seek_to(self.slider_seek.value())

	def on_seek_changed(self, val: int) -> None:
		if not self.is_video:
			return
		if self._is_scrubbing:
			self._seek_to(int(val))

	def _seek_to(self, idx: int) -> None:
		if not self.is_video or self.cap is None or not self.cap.isOpened():
			return
		try:
			idx = max(0, min(int(idx), max(0, self.total_frames - 1))) if self.total_frames > 0 else max(0, int(idx))
			self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
			ok, frame = self.cap.read()
			if ok:
				self.cur_frame_idx = idx + 1
				self.last_frame = frame
				self._display_frame(frame, processing=False)
				if self.total_frames > 0:
					self.slider_seek.blockSignals(True)
					self.slider_seek.setValue(idx)
					self.slider_seek.blockSignals(False)
		except Exception:
			pass


	def on_mode_changed(self, text: str) -> None:
		self.state.mode = text
		# Default models for convenience
		if text == "head":
			self.model_edit.setText("models/face_detection_yunet_2023mar.onnx")
			self._select_first_matching_model_suffix([".onnx"])
		elif text == "body":
			if "models/" in self.model_edit.text():
				self.model_edit.setText("models/yolov8n.pt")
			self._select_first_matching_model_suffix([".pt"]) 
		# Re-init detector if capture already opened
		if self.cap is not None and self.cap.isOpened():
			self._reinit_detector()
		# re-apply lines to engine
		try:
			if hasattr(self.engine, 'set_lines') and self.state.lines:
				self.engine.set_lines(self.state.lines)
			else:
				self.engine.set_line(self.video.get_line())
		except Exception:
			pass

	def _reinit_detector(self) -> None:
		self.engine.set_mode(self.state.mode)
		mpath = self.model_edit.text().strip()
		if self.state.mode == "head":
			if not self._ensure_model(Path(mpath)):
				return
		self.engine.init_detector(mpath, self.state.confidence, self.state.imgsz)
		# Apply lines after reinit
		try:
			if hasattr(self.engine, 'set_lines') and self.state.lines:
				self.engine.set_lines(self.state.lines)
			else:
				self.engine.set_line(self.video.get_line())
		except Exception:
			pass
		# Apply device selection for YOLO detectors
		try:
			dev = self.cmb_device.currentText().lower()
			if hasattr(self.engine.detector, 'set_device'):
				# Prefer explicit variant if selected
				variant = self.cmb_device_variant.currentText().strip()
				dev_str = variant if (dev == 'gpu' and variant.startswith('cuda')) else ('cuda' if dev == 'gpu' else 'cpu')
				self.engine.detector.set_device(dev_str)
				# Apply half toggle if supported
				if hasattr(self.engine.detector, 'set_half'):
					self.engine.detector.set_half(self.chk_half.isChecked())
		except Exception:
			pass

	def on_conf_changed(self, val: int) -> None:
		self.state.confidence = round(val / 100.0, 2)
		self.lbl_conf.setText(f"Guven: {self.state.confidence:.2f}")
		if self.engine.detector is not None:
			self.engine.detector.confidence = self.state.confidence

	def on_model_selected(self, text: str) -> None:
		self.model_edit.setText(text)
		if self.cap is not None and self.cap.isOpened():
			self._reinit_detector()

	def on_line_changed(self, line: tuple) -> None:
		# Update currently selected line in engine state
		try:
			idx = self.state.selected_line_idx
			if idx < 0:
				return
			if hasattr(self.engine, 'set_lines') and self.state.lines:
				self.state.lines[idx] = (int(line[0]), int(line[1]), int(line[2]), int(line[3]))
				self.engine.set_lines(self.state.lines)
			else:
				self.engine.set_line((int(line[0]), int(line[1]), int(line[2]), int(line[3])))
			self._sync_lines_list_widget()
		except Exception:
			pass

	def on_lines_changed(self, lines: list) -> None:
		# Replace all lines in state and engine
		try:
			self.state.lines = [(int(l[0]), int(l[1]), int(l[2]), int(l[3])) for l in lines]
			if not self.state.lines:
				self.state.selected_line_idx = -1
			else:
				self.state.selected_line_idx = min(self.state.selected_line_idx or 0, len(self.state.lines) - 1)
			if hasattr(self.engine, 'set_lines'):
				self.engine.set_lines(self.state.lines)
			self._sync_lines_list_widget()
		except Exception:
			pass

	def on_show_trails_toggled(self, checked: bool) -> None:
		self.engine.viz.show_trails = checked

	def on_device_changed(self, text: str) -> None:
		try:
			dev = 'cuda' if text.lower() == 'gpu' else 'cpu'
			# Populate device variant list
			self._refresh_device_variants(dev)
			self._update_device_controls_visibility()
			if self.engine.detector is not None and hasattr(self.engine.detector, 'set_device'):
				# If GPU selected but model is CPU-only, warn and force CPU
				if dev == 'cuda' and hasattr(self.engine.detector, 'is_gpu_capable') and not self.engine.detector.is_gpu_capable():
					QtWidgets.QMessageBox.information(self, "Model GPU Desteklemiyor", "Secilen model GPU tabanli degil (yalnizca CPU). Cihaz CPU'ya ayarlanacak.")
					dev = 'cpu'
					self._force_cpu_selection()
				variant = self.cmb_device_variant.currentText().strip()
				dev_str = variant if (dev == 'cuda' and variant.startswith('cuda')) else dev
				self.engine.detector.set_device(dev_str)
				# Verify
				if dev == 'cuda' and hasattr(self.engine.detector, 'get_device') and self.engine.detector.get_device() != 'cuda' and not self.engine.detector.get_device().startswith('cuda'):
					QtWidgets.QMessageBox.information(self, "CUDA Kullanilamiyor", "GPU secildi ancak CUDA kullanilamadi. CUDA destekli PyTorch ve suruculerin yüklü oldugundan emin olun. CPU'ya düsüldü.")
				# Apply half toggle if supported
				if hasattr(self.engine.detector, 'set_half'):
					self.engine.detector.set_half(self.chk_half.isChecked())
		except Exception:
			pass

	def on_device_variant_changed(self, text: str) -> None:
		try:
			if not text:
				return
			if self.engine.detector is not None and hasattr(self.engine.detector, 'set_device'):
				# If variant indicates cuda:* but model is cpu-only, warn and switch to CPU
				if text.lower().startswith('cuda') and hasattr(self.engine.detector, 'is_gpu_capable') and not self.engine.detector.is_gpu_capable():
					QtWidgets.QMessageBox.information(self, "Model GPU Desteklemiyor", "Secilen model GPU tabanli degil (yalnizca CPU). Cihaz CPU'ya ayarlanacak.")
					self._force_cpu_selection()
					self.engine.detector.set_device('cpu')
					return
				self.engine.detector.set_device(text)
				if hasattr(self.engine.detector, 'set_half'):
					self.engine.detector.set_half(self.chk_half.isChecked())
		except Exception:
			pass

	def on_half_toggled(self, checked: bool) -> None:
		try:
			if self.engine.detector is not None and hasattr(self.engine.detector, 'set_half'):
				self.engine.detector.set_half(bool(checked))
		except Exception:
			pass

	def _refresh_device_variants(self, base: str) -> None:
		self.cmb_device_variant.blockSignals(True)
		self.cmb_device_variant.clear()
		variants = ["cpu"]
		if base in ("cuda", "gpu"):
			try:
				import torch  # type: ignore
				if hasattr(torch, 'cuda') and torch.cuda.is_available():
					count = torch.cuda.device_count()
					for i in range(count):
						variants.append(f"cuda:{i}")
			except Exception:
				pass
		for v in variants:
			self.cmb_device_variant.addItem(v)
		# Select a sensible default
		idx = self.cmb_device_variant.findText('cuda:0') if base in ("cuda", "gpu") else self.cmb_device_variant.findText('cpu')
		if idx >= 0:
			self.cmb_device_variant.setCurrentIndex(idx)
		self.cmb_device_variant.blockSignals(False)
		# Also update visibility to match base
		self._update_device_controls_visibility()

	def _refresh_backends(self) -> None:
		# List available GPU-capable modules/backends
		modules = []
		try:
			import torch  # type: ignore
			modules.append(f"PyTorch CUDA: {'Yes' if (hasattr(torch,'cuda') and torch.cuda.is_available()) else 'No'}")
		except Exception:
			modules.append("PyTorch: Not available")
		try:
			import onnxruntime as ort  # type: ignore
			providers = []
			try:
				providers = list(getattr(ort, 'get_available_providers')())
			except Exception:
				providers = []
			gpu = any('CUDA' in p or 'Dml' in p or 'DirectML' in p for p in providers)
			modules.append(f"ONNXRuntime GPU: {'Yes' if gpu else 'No'}")
		except Exception:
			modules.append("ONNXRuntime: Not available")
		self.cmb_backend.clear()
		for m in modules:
			self.cmb_backend.addItem(m)

	def _scan_models_dir(self) -> list[str]:
		base = Path.cwd() / "models"
		if not base.exists() or not base.is_dir():
			return []
		items: list[str] = []
		for p in sorted(base.iterdir()):
			if p.is_file() and p.suffix.lower() in (".pt", ".onnx"):
				try:
					rel = p.relative_to(Path.cwd())
					items.append(str(rel))
				except Exception:
					items.append(str(p))
		return items

	def _populate_models_list(self) -> None:
		models = self._scan_models_dir()
		self.cmb_model.blockSignals(True)
		self.cmb_model.clear()
		for m in models:
			self.cmb_model.addItem(m)
		current = self.model_edit.text().strip()
		idx = self.cmb_model.findText(current)
		if idx >= 0:
			self.cmb_model.setCurrentIndex(idx)
		elif models:
			self.cmb_model.setCurrentIndex(0)
			self.model_edit.setText(self.cmb_model.currentText())
		self.cmb_model.blockSignals(False)

	def _sync_lines_list_widget(self) -> None:
		self.list_lines.blockSignals(True)
		self.list_lines.clear()
		for i, (x1, y1, x2, y2) in enumerate(self.video.get_lines()):
			self.list_lines.addItem(f"{i+1}) ({x1},{y1}) -> ({x2},{y2})")
		# select current
		idx = max(0, min(self.state.selected_line_idx or 0, self.list_lines.count() - 1)) if self.list_lines.count() > 0 else -1
		if idx >= 0:
			self.list_lines.setCurrentRow(idx)
		self.list_lines.blockSignals(False)

	def on_add_line(self) -> None:
		# add a default horizontal line around middle
		if self.frame_size is not None:
			w, h = self.frame_size
			new_line = (w // 4, h // 2, (3 * w) // 4, h // 2)
		else:
			new_line = (100, 200, 500, 200)
		self.video.add_line(new_line)
		self.state.lines = self.video.get_lines()
		self.state.selected_line_idx = len(self.state.lines) - 1
		try:
			if hasattr(self.engine, 'set_lines'):
				self.engine.set_lines(self.state.lines)
		except Exception:
			pass
		self._sync_lines_list_widget()

	def on_remove_line(self) -> None:
		row = self.list_lines.currentRow()
		if row < 0:
			return
		self.video.remove_line_at(row)
		self.state.lines = self.video.get_lines()
		self.state.selected_line_idx = min(row, len(self.state.lines) - 1)
		try:
			if hasattr(self.engine, 'set_lines'):
				self.engine.set_lines(self.state.lines)
		except Exception:
			pass
		self._sync_lines_list_widget()

	def _select_first_matching_model_suffix(self, suffixes: list[str]) -> None:
		for i in range(self.cmb_model.count()):
			text = self.cmb_model.itemText(i).lower()
			if any(text.endswith(s) for s in suffixes):
				self.cmb_model.setCurrentIndex(i)
				self.model_edit.setText(self.cmb_model.currentText())
				break

	def _ensure_model(self, path: Path) -> bool:
		if path.exists():
			return True
		# Attempt to download common face model automatically
		try:
			import requests
			if path.name.lower() == "yolov8n-face.pt":
				url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-face.pt"
				path.parent.mkdir(parents=True, exist_ok=True)
				r = requests.get(url, timeout=60)
				r.raise_for_status()
				path.write_bytes(r.content)
				return True
		except Exception:
			pass
		QtWidgets.QMessageBox.warning(self, "Model Eksik", f"Model bulunamadi: {path}. Lütfen dosyayi indirin veya yolu degistirin.")
		return False

	def on_open(self) -> None:
		# Initialize capture and detector
		category = self.cmb_source_category.currentIndex()
		if category == 0:
			# Camera
			src_text = self.cmb_camera_index.currentText().strip() or "0"
			try:
				source: str | int = int(src_text)
			except ValueError:
				source = 0
		elif category == 1:
			# Video
			source = self.cmb_video_list.currentText().strip()
			# Ensure any previous connecting overlay is hidden
			self._set_connecting(False)
		else:
			# RTSP
			source = self.edit_rtsp_url.text().strip()
		self.state.source = source
		# Show connecting overlay for camera/RTSP
		if category in (0, 2):
			self._set_connecting(True, "camera" if category == 0 else "rtsp")
		self.state.model_path = self.model_edit.text().strip()
		self.state.mode = self.cmb_mode.currentText()
		self.engine.set_mode(self.state.mode)
		model_ok = self._ensure_model(Path(self.state.model_path)) if self.state.mode == "head" else True
		if not model_ok:
			return
		self.engine.init_detector(self.state.model_path, self.state.confidence, self.state.imgsz)
		# Apply counting method selection
		try:
			self.on_count_method_changed(self.cmb_count_method.currentIndex())
		except Exception:
			pass
		# Set initial device
		try:
			dev = self.cmb_device.currentText().lower()
			self._refresh_device_variants('cuda' if dev == 'gpu' else 'cpu')
			self._refresh_backends()
			if hasattr(self.engine.detector, 'set_device'):
				# Warn if model is not GPU-capable but GPU selected
				if dev == 'gpu' and hasattr(self.engine.detector, 'is_gpu_capable') and not self.engine.detector.is_gpu_capable():
					QtWidgets.QMessageBox.information(self, "Model GPU Desteklemiyor", "Secilen model GPU tabanli degil (yalnizca CPU). Cihaz CPU'ya ayarlanacak.")
					dev = 'cpu'
					self._force_cpu_selection()
				variant = self.cmb_device_variant.currentText().strip()
				dev_str = variant if (dev == 'gpu' and variant.startswith('cuda')) else ('cuda' if dev == 'gpu' else 'cpu')
				self.engine.detector.set_device(dev_str)
				if dev == 'gpu' and hasattr(self.engine.detector, 'get_device') and self.engine.detector.get_device() != 'cuda' and not self.engine.detector.get_device().startswith('cuda'):
					QtWidgets.QMessageBox.information(self, "CUDA Kullanilamiyor", "GPU secildi ancak CUDA kullanilamadi. CUDA destekli PyTorch ve suruculerin yüklü oldugundan emin olun. CPU'ya düsüldü.")
		except Exception:
			pass
		# Stop any previous capture thread before reopening
		self._stop_capture_thread()
		# (re)open capture
		if self.cap is not None:
			self.cap.release()
		self.is_video = (category == 1) and self._is_video_path(self.state.source)
		if self.is_video:
			self.cap = open_capture(self.state.source, prefer_quality=False)
			if not self.cap.isOpened():
				self._set_connecting(False)
				QtWidgets.QMessageBox.critical(self, "Hata", f"Kaynak acilamadi: {self.state.source}")
				return
			cap_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
			self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or -1)
			# Determine playback fps
			# Establish base FPS and speed control
			self.base_fps = float(cap_fps) if cap_fps and cap_fps > 0 else 25.0
			self.fps = self.base_fps * self.state.video_speed_mult
			self.lbl_speed.setText(f"Hiz: {self.state.video_speed_mult:.2f}x")
			self.slider_speed.setValue(int(self.state.video_speed_mult * 100))
			self._update_video_controls_enabled(True)
		else:
			# Defer opening to background thread to avoid UI blocking
			self.cap = None
			self.total_frames = -1
			self.base_fps = 25.0
			self.fps = self.base_fps
			self._update_video_controls_enabled(False)
		# Set initial frame size for videos if available
		if self.is_video:
			w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			self.frame_size = (w, h)
			self.video.set_frame_size(w, h)
		# Auto load lines for this session
		self._auto_load_line()
		# Prepare initial frame behavior
		self.btn_toggle.setText("Taramayi Baslat")
		self.state.running = False
		self.cur_frame_idx = 0
		self.last_frame = None
		if self.is_video:
			# Load first frame and display paused
			self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			ok, frame0 = self.cap.read()
			if ok:
				self.cur_frame_idx = 1
				self.last_frame = frame0
				self._display_frame(frame0, processing=False)
			# Timer ticks at UI refresh rate while paused
			self.timer.start(33)
		else:
			# For camera/stream, start timer unthrottled, not running
			self._cap_reconnect_backoff_s = 2.0
			self._start_capture_thread()
			self.timer.start(0)
		# Update seek controls based on new source
		self._update_seek_controls()

	def on_toggle(self) -> None:
		if self.cap is None:
			self.on_open()
			if self.cap is None or not self.cap.isOpened():
				return
		self.state.running = not self.state.running
		self.btn_toggle.setText("Taramayi Durdur" if self.state.running else "Taramayi Baslat")
		# Adjust timer for video playback
		if self.is_video:
			if self.state.running:
				play_fps = max(1e-3, (self.base_fps if self.base_fps else 25.0) * max(0.1, self.state.video_speed_mult))
				interval = int(max(1, round(1000.0 / play_fps)))
				self.timer.setInterval(interval)
			else:
				self.timer.setInterval(33)
		# Start/stop background threads
		if self.state.running:
			self._start_worker_threads()
		else:
			self._stop_worker_threads()

	def on_record(self) -> None:
		if self.cap is None or self.frame_size is None:
			return
		self.state.recording = not self.state.recording
		if self.state.recording:
			Path("outputs").mkdir(parents=True, exist_ok=True)
			stamp = time.strftime("%Y%m%d_%H%M%S")
			path = f"outputs/record_{stamp}.mp4"
			w, h = self.frame_size
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			self.writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
			self._start_writer_thread()
			self.btn_record.setText("Kaydi Durdur")
		else:
			self._stop_writer_thread()
			self.btn_record.setText("Kaydi Baslat")

	def on_reset(self) -> None:
		self.engine.reset_counts()
		self.lbl_counts.setText("Giris: 0 | Cikis: 0")

	def _auto_load_line(self) -> None:
		try:
			default_path = Path.cwd() / "config" / "line.yaml"
			# First, attempt per-source lines
			if hasattr(self.state, 'source') and self.state.source is not None:
				lines_t = load_lines_for_source(default_path, self.state.source)
				if not lines_t:
					line_t = load_line_for_source(default_path, self.state.source)
					self.state.lines = [line_t]
				else:
					self.state.lines = lines_t
			else:
				# Fallback to global schema
				lines_t = load_lines(default_path)
				if not lines_t:
					line_t = load_line(default_path)
					self.state.lines = [line_t]
				else:
					self.state.lines = lines_t
			self.state.selected_line_idx = 0 if self.state.lines else -1
			self.video.set_lines(self.state.lines)
			if hasattr(self.engine, 'set_lines'):
				self.engine.set_lines(self.state.lines)
			else:
				self.engine.set_line(self.state.lines[0] if self.state.lines else (100,200,500,200))
			self._sync_lines_list_widget()
		except Exception:
			pass

	def on_save_line(self) -> None:
		default_path = Path.cwd() / "config" / "line.yaml"
		default_path.parent.mkdir(parents=True, exist_ok=True)
		try:
			lines = self.video.get_lines()
			if hasattr(self.state, 'source') and self.state.source is not None:
				# Save under per-source key primarily
				if len(lines) <= 1:
					save_line_for_source(default_path, self.state.source, lines[0] if lines else (100,200,500,200))
				else:
					save_lines_for_source(default_path, self.state.source, lines)
			else:
				# Fallback to global schema
				if len(lines) <= 1:
					save_line(default_path, lines[0] if lines else (100, 200, 500, 200))
				else:
					save_lines(default_path, lines)
		except Exception as e:
			QtWidgets.QMessageBox.critical(self, "Hata", f"Kaydetme basarisiz: {e}")

	def on_tick(self) -> None:
		# If a reconnect requested showing overlay, do it now
		if self._request_connecting_overlay and not self._is_connecting:
			reason = "rtsp" if (isinstance(self.state.source, str) and str(self.state.source).strip().lower().startswith("rtsp://")) else "camera"
			self._set_connecting(True, reason)
			self._request_connecting_overlay = False
		# Allow UI updates even if capture isn't ready; show connecting overlay timeout if needed
		if self.cap is None or not self.cap.isOpened():
			if self._is_connecting:
				self._maybe_show_connect_error_popup()
			return
		# Video vs camera handling
		if self.is_video:
			if self.state.running:
				ok, frame = self.cap.read()
				if not ok:
					return
				self.cur_frame_idx += 1
				self.last_frame = frame.copy()
				try:
					frame_out = self.engine.process(frame)
				except Exception as e:
					frame_out = frame
				else:
					cin, cout = self.engine.counts
					dev = ''
					try:
						if self.engine.detector is not None and hasattr(self.engine.detector, 'get_device'):
							dev = f" | Aygit: {self.engine.detector.get_device()}"
					except Exception:
						dev = ''
					self.lbl_counts.setText(f"Giris: {cin} | Cikis: {cout}{dev}")
				# display and optionally record
				self._display_frame(frame_out, processing=False)
				if self.state.recording and self.writer is not None:
					try:
						self._writer_queue.append(frame_out)
					except Exception:
						pass
				# update seek slider position
				if self.total_frames > 0:
					cur = max(0, self.cur_frame_idx - 1)
					self.slider_seek.blockSignals(True)
					self.slider_seek.setValue(cur)
					self.slider_seek.blockSignals(False)
			else:
				# Paused: show last frame without advancing
				if self.last_frame is not None:
					self._display_frame(self.last_frame.copy(), processing=False)
				return
		else:
			# Camera/stream: pull latest frame from background capture queue (non-blocking)
			frame = None
			try:
				item = self._cap_queue.popleft()
				frame, capture_ts = item
			except Exception:
				frame = None
			if frame is None:
				# No new frame; keep last frame on screen if available
				if self.last_frame is not None:
					self._display_frame(self.last_frame.copy(), processing=False)
				# if still connecting, check timeout and possibly show popup
				if self._is_connecting:
					self._maybe_show_connect_error_popup()
				return
			# we received a frame — stop connecting overlay if active
			if self._is_connecting:
				self._set_connecting(False)
			# Update metrics
			now = time.time()
			if self._cap_prev_ts is not None:
				delta = max(1e-6, capture_ts - self._cap_prev_ts)
				fps_inst = 1.0 / delta
				self._cap_fps_ewma = (0.9 * self._cap_fps_ewma) + (0.1 * fps_inst) if self._cap_fps_ewma > 0 else fps_inst
				jitter_ms = abs(delta - (1.0 / max(1e-6, self.base_fps))) * 1000.0
				self._cap_jitter_ms_ewma = (0.9 * self._cap_jitter_ms_ewma) + (0.1 * jitter_ms)
				# Approximate end-to-end latency: time since capture timestamp
				lat_ms = max(0.0, (now - capture_ts) * 1000.0)
				self._rtsp_latency_ms_ewma = (0.8 * self._rtsp_latency_ms_ewma) + (0.2 * lat_ms) if self._rtsp_latency_ms_ewma > 0 else lat_ms
			self._cap_prev_ts = capture_ts
			self._ui_frames_accum += 1
			if (now - self._ui_last_rate_time) >= 1.0:
				fps_ui = self._ui_frames_accum / max(1e-6, (now - self._ui_last_rate_time))
				self._ui_fps_ewma = (0.7 * self._ui_fps_ewma) + (0.3 * fps_ui) if self._ui_fps_ewma > 0 else fps_ui
				self._ui_frames_accum = 0
				self._ui_last_rate_time = now
			# Process if running
			if self.state.running:
				try:
					frame = self.engine.process(frame)
				except Exception as e:
					pass
				else:
					cin, cout = self.engine.counts
					dev = ''
					try:
						if self.engine.detector is not None and hasattr(self.engine.detector, 'get_device'):
							dev = f" | Aygit: {self.engine.detector.get_device()}"
					except Exception:
						dev = ''
					self.lbl_counts.setText(f"Giris: {cin} | Cikis: {cout}{dev}")
			# display and optionally record
			self.last_frame = frame
			self._display_frame(frame, processing=False)
			if self.state.recording and self.writer is not None:
				try:
					self._writer_queue.append(frame)
				except Exception:
					pass
			# Periodically update stream label
			if (time.time() - self._last_stream_label_update) >= 0.5:
				lat_ms = self._rtsp_latency_ms_ewma if self._rtsp_latency_ms_ewma > 0 else 0.0
				cap_fps = self._cap_fps_ewma if self._cap_fps_ewma > 0 else 0.0
				jit = self._cap_jitter_ms_ewma if self._cap_jitter_ms_ewma > 0 else 0.0
				ui_fps = self._ui_fps_ewma if self._ui_fps_ewma > 0 else 0.0
				self.lbl_stream.setText(f"Akis FPS: {cap_fps:.1f} | UI FPS: {ui_fps:.1f} | Jitter: {jit:.0f}ms | Gecikme: {lat_ms:.0f}ms")
				self._last_stream_label_update = time.time()

	def _set_connecting(self, active: bool, reason: Optional[str] = None) -> None:
		if active:
			self._is_connecting = True
			self._connecting_started = time.time()
			self._connect_popup_shown = False
			if reason:
				self._connecting_reason = reason
			msg = (
				"Kamera ile iletişime geçilmeye çalışılıyor..."
				if self._connecting_reason == "camera"
				else "RTSP bağlantısı kurulmaya çalışılıyor..."
			)
			self.overlay.set_text(msg)
			self.overlay.setGeometry(self.video.rect())
			self.overlay.start()
		else:
			self._is_connecting = False
			self.overlay.stop()

	def _maybe_show_connect_error_popup(self) -> None:
		# Show the standard error popup if connecting takes too long without any frames
		if self._connect_popup_shown:
			return
		elapsed = time.time() - self._connecting_started
		if elapsed >= 10.0:
			self._connect_popup_shown = True
			self.overlay.stop()
			# Stop capture attempts while showing error
			self._stop_capture_thread()
			if self.cap is not None:
				try:
					self.cap.release()
				except Exception:
					pass
			self.cap = None
			self._is_connecting = False
			# Show the same kind of popup content as before
			QtWidgets.QMessageBox.critical(self, "Hata", f"Kaynak acilamadi: {self.state.source}")

	def closeEvent(self, event: QtGui.QCloseEvent) -> None:
		# Persist current UI settings
		try:
			self._save_ui_config()
		except Exception:
			pass
		self.timer.stop()
		self._stop_worker_threads()
		self._stop_writer_thread()
		self._stop_capture_thread()
		if self.cap is not None:
			self.cap.release()
		if self.writer is not None:
			self.writer.release()
		return super().closeEvent(event)

	def on_count_method_changed(self, idx: int) -> None:
		try:
			key = self.cmb_count_method.currentData()
			if not key:
				return
			self.state.count_method = str(key)
			if hasattr(self.engine, 'set_counting_method'):
				self.engine.set_counting_method(self.state.count_method)
		except Exception:
			pass

	# -------- UI settings persistence --------
	def _load_ui_config(self) -> None:
		# Ensure config files exist, then load app.yaml and apply to UI
		ensure_default_configs()
		cfg_path = DEFAULT_APP_CFG
		try:
			with cfg_path.open("r", encoding="utf-8") as f:
				data = yaml.safe_load(f) or {}
		except Exception:
			data = {}
		# Source persistence for all categories
		# Primary 'source' (for backward compatibility)
		src = str(data.get("source", "0"))
		# Per-category saved values
		saved_cam = str(data.get("last_camera_index", "0"))
		saved_vid = str(data.get("last_video_path", src if self._is_video_path(src) else ""))
		saved_rtsp = str(data.get("last_rtsp_url", src if isinstance(src, str) and src.lower().startswith("rtsp://") else ""))
		# Apply to UI widgets regardless of selected category
		# Camera
		try:
			int(saved_cam)
			idx = self.cmb_camera_index.findText(saved_cam)
			if idx < 0:
				self.cmb_camera_index.addItem(saved_cam)
				idx = self.cmb_camera_index.findText(saved_cam)
			if idx >= 0:
				self.cmb_camera_index.setCurrentIndex(idx)
		except Exception:
			pass
		# Video
		if saved_vid:
			if self.cmb_video_list.findText(saved_vid) < 0:
				self.cmb_video_list.addItem(saved_vid)
			self.cmb_video_list.setCurrentText(saved_vid)
		# RTSP
		if saved_rtsp:
			self.edit_rtsp_url.setText(saved_rtsp)
		# Select category based on 'source' for initial view only
		cat = self._derive_category_from_source(src)
		self.cmb_source_category.setCurrentIndex(cat)
		self.source_stack.setCurrentIndex(cat)
		# Model
		model = str(data.get("model", self.model_edit.text().strip()))
		self.model_edit.setText(model)
		idx = self.cmb_model.findText(model)
		if idx >= 0:
			self.cmb_model.setCurrentIndex(idx)
		# Confidence
		try:
			conf = float(data.get("confidence", self.state.confidence))
			self.conf_slider.setValue(int(max(1, min(100, round(conf * 100)))))
		except Exception:
			pass
		# Mode
		mode = str(data.get("mode", self.state.mode))
		if mode in ("body", "head"):
			self.cmb_mode.setCurrentText(mode)
		# Counting method
		try:
			cm = str(data.get("count_method", self.state.count_method))
			# select matching userData
			for i in range(self.cmb_count_method.count()):
				if str(self.cmb_count_method.itemData(i)) == cm:
					self.cmb_count_method.setCurrentIndex(i)
					self.state.count_method = cm
					break
			# Apply to engine
			if hasattr(self.engine, 'set_counting_method'):
				self.engine.set_counting_method(self.state.count_method)
		except Exception:
			pass
		# Device and variant
		dev_raw = str(data.get("device", "cpu")).lower()
		dev_base = "gpu" if dev_raw.startswith("cuda") else "cpu"
		self.cmb_device.setCurrentText("GPU" if dev_base == "gpu" else "CPU")
		self._refresh_device_variants("cuda" if dev_base == "gpu" else "cpu")
		# Try to select exact variant if GPU
		if dev_raw.startswith("cuda"):
			idxv = self.cmb_device_variant.findText(dev_raw)
			if idxv >= 0:
				self.cmb_device_variant.setCurrentIndex(idxv)
		# Half precision
		try:
			self.chk_half.setChecked(bool(data.get("half", False)))
		except Exception:
			pass
		# ROI band
		try:
			val = int(data.get("line_band_px", 0))
			self.spin_roi_band.setValue(max(0, min(512, val)))
			self.on_roi_band_changed(self.spin_roi_band.value())
		except Exception:
			pass
		# Trails toggle
		try:
			show_tr = bool(data.get("show_trails", True))
			self.chk_show_trails.setChecked(show_tr)
		except Exception:
			pass
		# RTSP quality preference
		try:
			self.chk_rtsp_quality.setChecked(bool(data.get("rtsp_prefer_quality", False)))
		except Exception:
			pass
		# ROI overlay visibility
		try:
			self.chk_roi_show.setChecked(bool(data.get("roi_show", True)))
		except Exception:
			pass
		# Playback speed
		try:
			spd = float(data.get("video_speed_mult", self.state.video_speed_mult))
			self.slider_speed.setValue(int(max(10, min(1700, round(spd * 100)))))
			self.on_speed_changed(self.slider_speed.value())
		except Exception:
			pass
		# Recent sources list (optional)
		try:
			recent = data.get("recent_sources", [])
			if isinstance(recent, list):
				for s in recent:
					if isinstance(s, str) and self.cmb_source.findText(s) < 0:
						self.cmb_source.addItem(s)
		except Exception:
			pass

	def _save_ui_config(self) -> None:
		cfg_path = DEFAULT_APP_CFG
		cfg_path.parent.mkdir(parents=True, exist_ok=True)
		data = {}
		# Load existing to preserve unknown fields
		try:
			with cfg_path.open("r", encoding="utf-8") as f:
				data = yaml.safe_load(f) or {}
		except Exception:
			data = {}
		# Update with UI state
		# Persist source values for all categories to avoid forgetting
		cam_val = (self.cmb_camera_index.currentText().strip() or "0")
		vid_val = self.cmb_video_list.currentText().strip()
		rtsp_val = self.edit_rtsp_url.text().strip()
		# Keep 'source' as the currently selected one (for compatibility)
		category = self.cmb_source_category.currentIndex()
		if category == 0:
			data["source"] = cam_val
		elif category == 1:
			data["source"] = vid_val
		else:
			data["source"] = rtsp_val
		# Save per-category remembered values
		data["last_camera_index"] = cam_val
		if vid_val:
			data["last_video_path"] = vid_val
		if rtsp_val:
			data["last_rtsp_url"] = rtsp_val
		data["model"] = self.model_edit.text().strip()
		data["confidence"] = round(self.state.confidence if hasattr(self, 'state') else (self.conf_slider.value() / 100.0), 2)
		# Device: prefer exact variant if GPU
		dev_sel = self.cmb_device.currentText().lower()
		if dev_sel == "gpu":
			variant = self.cmb_device_variant.currentText().strip() or "cuda"
			data["device"] = variant
		else:
			data["device"] = "cpu"
		data["half"] = bool(self.chk_half.isChecked())
		data["mode"] = self.cmb_mode.currentText().strip()
		data["line_band_px"] = int(self.spin_roi_band.value())
		data["count_method"] = str(self.state.count_method)
		data["show_trails"] = bool(self.chk_show_trails.isChecked())
		data["rtsp_prefer_quality"] = bool(self.chk_rtsp_quality.isChecked())
		data["roi_show"] = bool(self.chk_roi_show.isChecked())
		data["video_speed_mult"] = float(self.state.video_speed_mult)
		# Keep recent videos list (optional)
		recent_videos: list[str] = []
		for i in range(self.cmb_video_list.count()):
			text = self.cmb_video_list.itemText(i).strip()
			if text and text not in recent_videos:
				recent_videos.append(text)
		data["recent_videos"] = recent_videos[:20]
		try:
			with cfg_path.open("w", encoding="utf-8") as f:
				yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
		except Exception:
			pass

	# -------- Background capture thread --------
	def _start_capture_thread(self) -> None:
		if self._cap_thread is None or not self._cap_thread.is_alive():
			self._cap_stop.clear()
			self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
			self._cap_thread.start()

	def _stop_capture_thread(self) -> None:
		self._cap_stop.set()
		thr = self._cap_thread
		self._cap_thread = None
		if thr is not None:
			try:
				thr.join(timeout=0.5)
			except Exception:
				pass
		try:
			self._cap_queue.clear()
		except Exception:
			pass

	def _capture_loop(self) -> None:
		fail_count = 0
		while not self._cap_stop.is_set():
			try:
				if self.cap is None or not self.cap.isOpened():
					try:
						self.cap = open_capture(self.state.source, prefer_quality=self.chk_rtsp_quality.isChecked())
					except Exception:
						pass
					time.sleep(0.1)
					continue
				ok, fr = self.cap.read()
				if ok and fr is not None:
					fail_count = 0
					self._cap_reconnect_backoff_s = 2.0
					ts = time.time()
					try:
						self._cap_queue.append((fr, ts))
					except Exception:
						pass
					continue
				fail_count += 1
				if fail_count >= 15:
					try:
						self.cap.release()
					except Exception:
						pass
					self.cap = None
					# Ask GUI to show connecting overlay
					self._request_connecting_overlay = True
					time.sleep(min(10.0, max(0.2, self._cap_reconnect_backoff_s)))
					self._cap_reconnect_backoff_s = min(10.0, max(2.0, self._cap_reconnect_backoff_s * 1.5))
				else:
					time.sleep(0.02)
			except Exception:
				fail_count += 1
				time.sleep(0.1)

	# -------- Background processing threads --------
	def _enqueue_for_processing(self, frame: np.ndarray) -> None:
		if frame is None:
			return
		try:
			self._proc_queue.append(frame.copy())
		except Exception:
			pass

	def _start_worker_threads(self) -> None:
		if self._proc_thread is None or not self._proc_thread.is_alive():
			self._proc_stop.clear()
			self._proc_thread = threading.Thread(target=self._worker_loop, daemon=True)
			self._proc_thread.start()

	def _stop_worker_threads(self) -> None:
		self._proc_stop.set()
		thr = self._proc_thread
		self._proc_thread = None
		if thr is not None:
			try:
				thr.join(timeout=0.2)
			except Exception:
				pass

	def _worker_loop(self) -> None:
		while not self._proc_stop.is_set():
			fr = None
			try:
				fr = self._proc_queue.popleft()
			except Exception:
				fr = None
			if fr is None:
				time.sleep(0.005)
				continue
			try:
				out = self.engine.process(fr)
			except Exception:
				out = fr
			# update counts label safely on GUI thread
			cin, cout = self.engine.counts
			dev = ''
			try:
				if self.engine.detector is not None and hasattr(self.engine.detector, 'get_device'):
					dev = f" | Aygit: {self.engine.detector.get_device()}"
			except Exception:
				dev = ''
			QtCore.QMetaObject.invokeMethod(self.lbl_counts, "setText", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, f"Giris: {cin} | Cikis: {cout}{dev}"))
			# do not write from background processing thread; main thread handles writing

	# -------- Async writer thread --------
	def _start_writer_thread(self) -> None:
		if self._writer_thread is None or not self._writer_thread.is_alive():
			self._writer_stop.clear()
			self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
			self._writer_thread.start()

	def _stop_writer_thread(self) -> None:
		self._writer_stop.set()
		thr = self._writer_thread
		self._writer_thread = None
		if thr is not None:
			try:
				thr.join(timeout=0.2)
			except Exception:
				pass
		if self.writer is not None:
			try:
				self.writer.release()
			except Exception:
				pass
		self.writer = None

	def _writer_loop(self) -> None:
		while not self._writer_stop.is_set():
			fr = None
			try:
				fr = self._writer_queue.popleft()
			except Exception:
				fr = None
			if fr is None:
				time.sleep(0.01)
				continue
			try:
				if self.writer is not None:
					self.writer.write(fr)
			except Exception:
				pass


def main() -> int:
	app = QtWidgets.QApplication(sys.argv)
	w = MainWindow()
	w.show()
	return app.exec()


if __name__ == "__main__":
	sys.exit(main())
