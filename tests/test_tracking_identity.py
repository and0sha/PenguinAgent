"""Boundary association checks without NumPy, SciPy, models or video files."""

import copy
import unittest

from tools.tracking_identity import match_boundary


def history(*points):
    return [{'second': second, 'x': x, 'y': y} for second, x, y in points]


class TrackingIdentityTests(unittest.TestCase):
    def test_motion_prediction_preserves_crossing_tracks_instead_of_raw_ids(self):
        previous = {
            10: history((0, 0, 0), (1, 4, 0)),
            20: history((0, 10, 0), (1, 6, 0)),
        }
        current = {10: (2, 0), 20: (8, 0)}
        self.assertEqual(match_boundary(previous, current, 2, 10), {10: 20, 20: 10})

    def test_prediction_uses_seconds_and_both_axes(self):
        previous = {7: history((0, 0, 0), (2, 4, 6))}
        self.assertEqual(match_boundary(previous, {99: (6, 9)}, 3, 1), {99: 7})

    def test_single_observation_falls_back_to_last_centroid(self):
        self.assertEqual(match_boundary({1: history((1, 10, 10))}, {2: (10.5, 10)}, 2, 1), {2: 1})

    def test_exact_crossing_is_ambiguous_and_does_not_claim_identity(self):
        previous = {1: history((0, 0, 0), (1, 2, 0)), 2: history((0, 8, 0), (1, 6, 0))}
        self.assertEqual(match_boundary(previous, {10: (4, 0), 20: (4, 0)}, 2, 2), {})

    def test_two_current_points_competing_for_one_previous_id_cannot_both_match(self):
        previous = {1: history((0, 0, 0))}
        self.assertEqual(match_boundary(previous, {10: (0, 0), 20: (4, 0)}, 1, 10), {10: 1})

    def test_near_competition_on_previous_side_rejects_both_points(self):
        previous = {1: history((0, 0, 0))}
        self.assertEqual(match_boundary(previous, {10: (0.1, 0), 20: (0.2, 0)}, 1, 10), {})

    def test_exact_nearest_tie_is_unmatched(self):
        previous = {1: history((0, 0, 0)), 2: history((0, 2, 0))}
        self.assertEqual(match_boundary(previous, {99: (1, 0)}, 1, 5), {})

    def test_near_tie_on_current_side_is_unmatched(self):
        previous = {1: history((0, 0, 0)), 2: history((0, 0.4, 0))}
        self.assertEqual(match_boundary(previous, {99: (0.1, 0)}, 1, 10), {})

    def test_large_gate_ambiguity_margin_is_capped_at_two_pixels(self):
        previous = {1: history((0, 0, 0)), 2: history((0, 5, 0))}
        self.assertEqual(match_boundary(previous, {99: (0, 0)}, 1, 128), {99: 1})
        previous[2] = history((0, 2, 0))
        self.assertEqual(match_boundary(previous, {99: (0, 0)}, 1, 128), {})

    def test_small_gate_ambiguity_margin_scales_with_resolution(self):
        previous = {1: history((0, 0, 0)), 2: history((0, 0.08, 0))}
        self.assertEqual(match_boundary(previous, {99: (0, 0)}, 1, 0.5), {99: 1})
        previous[2] = history((0, 0.04, 0))
        self.assertEqual(match_boundary(previous, {99: (0, 0)}, 1, 0.5), {})

    def test_gate_filters_invalid_edges_before_assignment(self):
        # A full unconstrained minimum-cost assignment would choose 100->1 and
        # 200->2, then lose the second edge to the gate. The supported nearest
        # association is 100->2; the distant arrival must remain unmatched.
        previous = {1: history((0, 0, 0)), 2: history((0, 10, 0))}
        self.assertEqual(match_boundary(previous, {100: (6, 0), 200: (21, 0)}, 1, 6), {100: 2})

    def test_new_arrivals_and_reused_raw_ids_outside_gate_stay_unmatched(self):
        previous = {1: history((0, 0, 0))}
        self.assertEqual(match_boundary(previous, {2: (0.1, 0), 1: (100, 0)}, 1, 5), {2: 1})

    def test_gate_is_inclusive(self):
        self.assertEqual(match_boundary({0: history((0, 0, 0))}, {0: (3, 4)}, 1, 5), {0: 0})
        self.assertEqual(match_boundary({0: history((0, 0, 0))}, {0: (3, 4)}, 1, 4.99), {})

    def test_input_order_does_not_change_matching_and_outputs_are_injective(self):
        previous = {2: history((0, 10, 0)), 1: history((0, 0, 0))}
        current = {20: (10, 0), 10: (0, 0)}
        expected = {10: 1, 20: 2}
        for old_order in [previous, dict(reversed(list(previous.items())))]:
            for new_order in [current, dict(reversed(list(current.items())))]:
                result = match_boundary(old_order, new_order, 1, 4)
                self.assertEqual(result, expected)
                self.assertEqual(list(result), sorted(result))
                self.assertEqual(len(set(result.values())), len(result))

    def test_inputs_are_not_modified(self):
        previous = {1: history((0, 0, 0), (1, 1, 1))}
        current = {3: (2, 2)}
        original = copy.deepcopy((previous, current))
        match_boundary(previous, current, 2, 5)
        self.assertEqual((previous, current), original)

    def test_empty_inputs_have_no_identity_matches(self):
        self.assertEqual(match_boundary({}, {1: (0, 0)}, 1, 1), {})
        self.assertEqual(match_boundary({1: history((0, 0, 0))}, {}, 1, 1), {})
        self.assertEqual(match_boundary({}, {}, 0, 1), {})

    def test_nonfinite_numbers_and_invalid_types_are_rejected(self):
        valid = {1: history((0, 0, 0))}
        for invalid in [float('nan'), float('inf'), -float('inf'), True, '1', None]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    match_boundary(valid, {2: (invalid, 0)}, 1, 1)
                with self.assertRaises(ValueError):
                    match_boundary(valid, {2: (0, 0)}, invalid, 1)
                with self.assertRaises(ValueError):
                    match_boundary(valid, {2: (0, 0)}, 1, invalid)
                with self.assertRaises(ValueError):
                    match_boundary({1: history((0, invalid, 0))}, {2: (0, 0)}, 1, 1)

    def test_invalid_ids_and_container_shapes_raise_value_error(self):
        for invalid_id in [-1, True, '1', 1.5]:
            with self.subTest(invalid_id=invalid_id):
                with self.assertRaises(ValueError):
                    match_boundary({invalid_id: history((0, 0, 0))}, {}, 1, 1)
                with self.assertRaises(ValueError):
                    match_boundary({}, {invalid_id: (0, 0)}, 1, 1)
        invalid_cases = [([], {}, 1, 1), ({}, [], 1, 1), ({1: []}, {}, 1, 1),
                         ({1: [{'second': 0, 'x': 0}]}, {}, 1, 1),
                         ({}, {1: (0, 0, 0)}, 1, 1), ({}, {1: {'x': 0, 'y': 0}}, 1, 1)]
        for arguments in invalid_cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    match_boundary(*arguments)

    def test_nonpositive_gate_and_invalid_elapsed_times_are_rejected(self):
        for gate in [0, -1]:
            with self.assertRaises(ValueError):
                match_boundary({}, {}, 1, gate)
        for previous, current_second in [
            ({1: history((0, 0, 0))}, 0),
            ({1: history((2, 0, 0))}, 1),
            ({1: history((0, 0, 0), (0, 1, 1))}, 1),
            ({1: history((1, 0, 0), (0, 1, 1))}, 2),
            ({1: history((-1, 0, 0))}, 1),
        ]:
            with self.subTest(previous=previous, current_second=current_second):
                with self.assertRaises(ValueError):
                    match_boundary(previous, {}, current_second, 1)

    def test_overflowing_motion_prediction_is_rejected(self):
        previous = {1: history((0, 0, 0), (1e-320, 1, 1))}
        with self.assertRaisesRegex(ValueError, 'prediction'):
            match_boundary(previous, {2: (0, 0)}, 1, 1)


if __name__ == '__main__':
    unittest.main()
