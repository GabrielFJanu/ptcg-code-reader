import unittest
from math import ceil

import cv2
import numpy as np

from frames.processing import crop_card


class CropCardTests(unittest.TestCase):
    def test_visible_cards_near_each_edge_keep_their_full_extent(self):
        for center, size in (
            ((20, 100), (100, 20)),
            ((180, 100), (100, 20)),
            ((100, 20), (20, 100)),
            ((100, 180), (20, 100)),
        ):
            with self.subTest(center=center):
                polygon = cv2.boxPoints((center, size, 90))
                frame = np.zeros((200, 200, 3), dtype=np.uint8)
                cv2.fillConvexPoly(frame, polygon.astype(np.int32), (255, 255, 255))
                crop = crop_card(frame, polygon)
                self.assertEqual(sorted(crop.shape[:2]), [20, 100])
                self.assertGreater(crop.min(), 240)

    def test_rotated_cards_preserve_center_and_dimensions(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 4, (0, 255, 0), -1)
        for angle in (0, 15, 45, 75, 90, 135):
            with self.subTest(angle=angle):
                polygon = cv2.boxPoints(((100, 100), (80, 30), angle))
                _, (width, height), _ = cv2.minAreaRect(polygon)
                crop = crop_card(frame, polygon)
                self.assertEqual(crop.shape[:2], (ceil(height), ceil(width)))
                np.testing.assert_array_equal(
                    crop[crop.shape[0] // 2, crop.shape[1] // 2], [0, 255, 0]
                )

    def test_outside_pixels_are_padded_without_wrapping(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        polygon = cv2.boxPoints(((0, 50), (40, 20), 0))
        crop = crop_card(frame, polygon)
        self.assertEqual(sorted(crop.shape[:2]), [20, 40])
        self.assertEqual(crop.min(), 0)
        self.assertEqual(crop.max(), 255)
        self.assertGreater(np.mean(crop == 0), 0.4)

    def test_degenerate_polygons_return_none(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        for points in ([], [(1, 1)], [(1, 1), (2, 2)], [(1, 1)] * 3):
            with self.subTest(points=points):
                self.assertIsNone(crop_card(frame, np.array(points, dtype=np.float32)))


if __name__ == '__main__':
    unittest.main()
