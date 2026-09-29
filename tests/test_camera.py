import time
import unittest

import camera
import line_sim


class CameraFallbackTests(unittest.TestCase):
    def frame(self, command):
        # generate_frame reads the wall clock, so the state must be created against it too.
        now = time.time()
        return camera.generate_frame(line_sim.apply_command(line_sim.new_state(now), command, now))

    def test_frame_has_expected_shape(self):
        frame = self.frame("START")
        self.assertEqual(frame.shape, (720, 1280, 3))

    def test_paused_line_is_dimmed_by_overlay(self):
        running = self.frame("START")
        paused = self.frame("PAUSE")
        self.assertLess(paused.mean(), running.mean())


if __name__ == "__main__":
    unittest.main()
