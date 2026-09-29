"""Fallback 2D camera feed (OpenCV) for browsers without WebGL."""
import time

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
    frame = np.full((h, w, 3), (20, 25, 30), dtype=np.uint8)
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


def gen(state):
    while True:
        frame = generate_frame(state)
        (flag, encodedImage) = cv2.imencode('.jpg', frame)
        if not flag: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encodedImage.tobytes() + b'\r\n')
        time.sleep(0.03)
