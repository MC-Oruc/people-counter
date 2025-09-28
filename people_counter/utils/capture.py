from __future__ import annotations

import os
from typing import Union

import cv2


def _to_source(value: Union[int, str]) -> Union[int, str]:
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except Exception:
        return str(value)


def is_rtsp_url(source: Union[int, str]) -> bool:
    try:
        if isinstance(source, str) and source.strip().lower().startswith("rtsp://"):
            return True
    except Exception:
        pass
    return False


def open_capture(source: Union[int, str], prefer_quality: bool = False) -> cv2.VideoCapture:
    """Open a cv2.VideoCapture with sane defaults for RTSP.

    - For RTSP URLs, prefer FFMPEG backend with TCP transport and lower latency.
    - For others, fall back to default backend.
    """
    src = _to_source(source)

    if is_rtsp_url(src):
        prev_opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        # Prefer TCP to avoid UDP packet loss; tune buffer and max_delay (microseconds)
        if prefer_quality:
            # Larger timeouts and buffers to favor quality (accept higher latency)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|stimeout;10000000|rw_timeout;10000000|"
                "max_delay;15000000|buffer_size;10485760|probesize;5000000|analyzeduration;5000000"
            )
        else:
            # Lower buffering for lower latency
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|stimeout;5000000|rw_timeout;5000000|max_delay;5000000|buffer_size;102400"
            )
        try:
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        finally:
            # Restore previous env to avoid side effects for non-RTSP sources opened later
            if prev_opts is None:
                try:
                    del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
                except Exception:
                    pass
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev_opts
        # Adjust internal queueing based on preference
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 8 if prefer_quality else 1)
        except Exception:
            pass
        return cap

    # Non-RTSP: default behavior
    cap = cv2.VideoCapture(src)
    try:
        # If live camera index, reduce buffering
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap



