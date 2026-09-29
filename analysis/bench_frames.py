"""Latency (p50/p95) and throughput of the frame pipeline: read + JPEG encode.

    python analysis/bench_frames.py                        # simulated line
    python analysis/bench_frames.py video:/path/clip.mp4   # any OpenCV-readable video

Inference latency is not included yet; run the ONNX model on the same frames to add it.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import line_sim  # noqa: E402
from camera import make_source, measure_latency  # noqa: E402

if __name__ == "__main__":
    now = time.time()
    state = line_sim.apply_command(line_sim.new_state(now), "START", now)
    spec = sys.argv[1] if len(sys.argv) > 1 else None
    print(json.dumps({"source": spec or "simulated", **measure_latency(make_source(spec, state), frames=200)}, indent=2))
