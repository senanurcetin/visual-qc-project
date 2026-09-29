"""Camera feed: frame sources (simulated line, video file) and the MJPEG stream built on them.

The simulated source is also the 2D fallback for browsers without WebGL.
"""
import statistics
import time
from collections.abc import Iterator
from typing import Protocol

import cv2
import numpy as np

import line_sim

ANIMATION_CYCLE = line_sim.CYCLE_SECONDS


def generate_frame(state):
    """Yedek 2D kamera görüntüsü (WebGL olmayan tarayıcılar için OpenCV çizimi)."""
    snap = line_sim.snapshot(state, time.time())
    progress = (snap["sim_time"] % ANIMATION_CYCLE) / ANIMATION_CYCLE
    current_mode = snap["system_mode"]
    current_status = snap["current_unit_status"]

    # Görsel Çizim (OpenCV)
    w, h = 1280, 720
    frame: np.ndarray = np.full((h, w, 3), (20, 25, 30), dtype=np.uint8)
    cv2.rectangle(frame, (0, h//2 - 130), (w, h//2 + 130), (40, 45, 50), -1)
    prod_x = int(w + 100 - (progress * (w + 400)))

    if -200 < prod_x < w:
        if current_mode == 'RUNNING' and current_status != "PENDING":
            is_ok = current_status == "OK"
            color = (129, 185, 16) if is_ok else (68, 68, 239)
            cv2.rectangle(frame, (prod_x, h//2-150), (prod_x+160, h//2+150), (160, 165, 170), -1)
            cv2.rectangle(frame, (prod_x+10, h//2-140), (prod_x+150, h//2+140), (10, 10, 10), -1)
            cv2.rectangle(frame, (prod_x-5, h//2-155), (prod_x+165, h//2+155), color, 2)
            cv2.putText(frame, f"{current_status}", (prod_x+20, h//2+10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, "CAM_01 > QC_SCAN", (prod_x, h//2-175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, f"MATCH: {99.8 if is_ok else 42.1}%", (prod_x, h//2+175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if current_mode != 'RUNNING':
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        msg = "SYSTEM " + current_mode
        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        cv2.putText(frame, msg, ((w - text_size[0]) // 2, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

    return frame


class FrameSource(Protocol):
    """Anything that yields BGR frames: the simulator, a video file, later a real camera."""

    def read(self) -> np.ndarray: ...


class SimulatedSource:
    def __init__(self, state):
        self.state = state

    def read(self) -> np.ndarray:
        return generate_frame(self.state)


class VideoFileSource:
    """Loops a video file, e.g. a recording of NEU-CLS plates passing the camera."""

    def __init__(self, path: str):
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise ValueError(f"cannot open video source: {path}")

    def read(self) -> np.ndarray:
        ok, frame = self.capture.read()
        if not ok:  # end of file: start over
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
            if not ok:
                raise ValueError("video source produced no frames")
        return frame

    def __del__(self):
        capture = getattr(self, "capture", None)
        if capture is not None:
            capture.release()


def make_source(spec: str | None, state) -> FrameSource:
    """`FRAME_SOURCE` value -> source. `video:<path>` plays a file; anything else is the simulator."""
    if spec and spec.startswith("video:"):
        try:
            return VideoFileSource(spec.removeprefix("video:"))
        except ValueError:
            pass  # fall back rather than break the page
    return SimulatedSource(state)


def gen(source: FrameSource) -> Iterator[bytes]:
    while True:
        ok, encoded = cv2.imencode('.jpg', source.read())
        if not ok:
            continue
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded.tobytes() + b'\r\n'
        time.sleep(0.03)


def measure_latency(source: FrameSource, frames: int = 100) -> dict:
    """Per-frame read + JPEG-encode time in milliseconds, and the sustainable frame rate."""
    times = []
    for _ in range(frames):
        start = time.perf_counter()
        cv2.imencode('.jpg', source.read())
        times.append((time.perf_counter() - start) * 1000)
    ordered = sorted(times)
    return {
        "frames": frames,
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[min(frames - 1, int(0.95 * frames))], 3),
        "max_ms": round(ordered[-1], 3),
        "fps": round(1000 / statistics.mean(times), 1),
    }
