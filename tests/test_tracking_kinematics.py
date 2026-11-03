"""Boundary velocity regression tests without models, OpenCV or a GPU."""

import copy
import unittest

from tools.tracking_kinematics import recompute_velocities
from tools.tracking_results import TrackingError


def point(second, x, y, vx=0, vy=0):
    return {"second": second, "x": x, "y": y, "vx": vx, "vy": vy, "area": 5}


class TrackingKinematicsTests(unittest.TestCase):
    def test_chunk_boundary_continuation_does_not_reset_velocity(self):
        # The local worker used to restart its velocity at second 10.
        tracks = {"penguin_1": [point(8, 16, 24, 2, 3), point(9, 18, 27, 2, 3), point(10, 20, 30)]}
        computed = recompute_velocities(tracks)["penguin_1"]
        self.assertEqual((computed[0]["vx"], computed[0]["vy"]), (0, 0))
        self.assertEqual((computed[1]["vx"], computed[1]["vy"]), (2, 3))
        self.assertEqual((computed[2]["vx"], computed[2]["vy"]), (2, 3))

    def test_noninteger_fps_and_short_chunk_use_actual_elapsed_time(self):
        # Proxy fps=2.5; a short first chunk has three frames (duration 1.2s).
        computed = recompute_velocities({"a": [point(0, 0, 0), point(.4, 1, 2), point(.8, 2, 4), point(1.2, 3, 6)]})["a"]
        for observation in computed[1:]:
            self.assertAlmostEqual(observation["vx"], 2.5)
            self.assertAlmostEqual(observation["vy"], 5)

    def test_observation_gap_produces_mean_velocity_over_that_interval(self):
        computed = recompute_velocities({"a": [point(0, 1, 10), point(5, 11, 0)]})["a"]
        self.assertEqual((computed[1]["vx"], computed[1]["vy"]), (2, -2))

    def test_new_identity_never_uses_another_tracks_previous_position(self):
        computed = recompute_velocities({"a": [point(9, 100, 50), point(10, 110, 60)], "b": [point(10, 0, 0)]})
        self.assertEqual(computed["a"][-1]["vx"], 10)
        self.assertEqual((computed["b"][0]["vx"], computed["b"][0]["vy"]), (0, 0))

    def test_stale_worker_velocity_is_recomputed_without_mutating_inputs(self):
        tracks = {"a": [point(0, 0, 0, 999, 999), point(2, 3, 4, 999, 999)]}
        original = copy.deepcopy(tracks)
        computed = recompute_velocities(tracks)
        self.assertEqual(computed["a"][1]["vx"], 1.5)
        self.assertEqual(computed["a"][1]["vy"], 2)
        self.assertEqual(tracks, original)
        self.assertIsNot(computed["a"][0], tracks["a"][0])

    def test_empty_scene_and_single_observation_are_valid(self):
        self.assertEqual(recompute_velocities({}), {})
        self.assertEqual(recompute_velocities({"a": [point(100, 1, 2)]})["a"][0]["vx"], 0)

    def test_duplicate_or_decreasing_timestamps_are_not_hidden_by_sorting(self):
        for timestamps in [(1, 1), (2, 1)]:
            with self.subTest(timestamps=timestamps), self.assertRaises(TrackingError):
                recompute_velocities({"a": [point(timestamps[0], 0, 0), point(timestamps[1], 1, 1)]})

    def test_invalid_input_measurements_and_derived_overflow_fail(self):
        cases = [
            [point(0, float("nan"), 0), point(1, 1, 1)],
            [point(0, 0, 0), point(float("inf"), 1, 1)],
            [point(0, False, 0), point(1, 1, 1)],
            [point(0, 0, 0), point(1e-320, 1, 1)],
            [point(0, -1e308, 0), point(1, 1e308, 0)],
        ]
        for observations in cases:
            with self.subTest(observations=observations), self.assertRaises(TrackingError):
                recompute_velocities({"a": observations})


if __name__ == "__main__":
    unittest.main()
