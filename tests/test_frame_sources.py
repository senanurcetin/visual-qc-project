import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

import camera
import line_sim


def write_video(path, frames=5, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), 40 * (i + 1), dtype=np.uint8))
    writer.release()


class FrameSourceTests(unittest.TestCase):
    def test_simulated_source_matches_generate_frame_shape(self):
        now = time.time()
        source = camera.make_source(None, line_sim.apply_command(line_sim.new_state(now), "START", now))
        self.assertIsInstance(source, camera.SimulatedSource)
        self.assertEqual(source.read().shape, (720, 1280, 3))

    def test_video_source_loops_at_end_of_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.avi"
            write_video(path, frames=4)
            source = camera.make_source(f"video:{path}", None)
            self.assertIsInstance(source, camera.VideoFileSource)
            brightness = [int(source.read().mean()) for _ in range(9)]  # 4 frames, read 9 times
            self.assertGreater(len(set(brightness)), 1)
            self.assertEqual(brightness[0], brightness[4])  # wrapped around to the first frame
            self.assertEqual(brightness[1], brightness[5])

    def test_unreadable_video_falls_back_to_the_simulator(self):
        now = time.time()
        state = line_sim.apply_command(line_sim.new_state(now), "START", now)
        self.assertIsInstance(camera.make_source("video:/nonexistent/clip.avi", state), camera.SimulatedSource)
        with self.assertRaises(ValueError):
            camera.VideoFileSource("/nonexistent/clip.avi")

    def test_mjpeg_generator_yields_jpeg_parts(self):
        now = time.time()
        part = next(camera.gen(camera.make_source(None, line_sim.new_state(now))))
        self.assertTrue(part.startswith(b"--frame\r\nContent-Type: image/jpeg"))
        self.assertIn(b"\xff\xd8", part)  # JPEG start-of-image marker

    def test_latency_report(self):
        now = time.time()
        report = camera.measure_latency(camera.make_source(None, line_sim.new_state(now)), frames=20)
        self.assertEqual(report["frames"], 20)
        self.assertLessEqual(report["p50_ms"], report["p95_ms"])
        self.assertLessEqual(report["p95_ms"], report["max_ms"])
        self.assertGreater(report["fps"], 0)


if __name__ == "__main__":
    unittest.main()
