import unittest

import polars as pl

from eye_tracking.columns import Column
from eye_tracking.fuse_eyes import EyeFusionProtocol, fuse_raw_data


class FuseRawEyeDataTest(unittest.TestCase):
    def test_fuse_drops_rows_with_both_eyes_missing_and_averages_the_rest(self):
        df = pl.DataFrame(
            {
                Column.Timestamp: [0, 10, 20, 30],
                Column.LeftX: [100.0, 0.0, 0.0, 300.0],
                Column.LeftY: [200.0, 0.0, 0.0, 400.0],
                Column.RightX: [110.0, 150.0, 0.0, 0.0],
                Column.RightY: [210.0, 250.0, 0.0, 0.0],
                Column.LeftDiameter: [3.0, 0.0, 0.0, 3.0],
                Column.RightDiameter: [3.2, 3.5, 0.0, 0.0],
            }
        )

        fused = fuse_raw_data(df)

        # the fully-missing row (timestamp=20) should be dropped entirely
        self.assertEqual(fused[Column.Timestamp].to_list(), [0, 10, 30])

        # row 0: both eyes valid -> averaged
        self.assertAlmostEqual(fused[Column.EyeX][0], 105.0)
        self.assertAlmostEqual(fused[Column.EyeY][0], 205.0)
        self.assertAlmostEqual(fused[Column.Diameter][0], 3.1)

        # row 1 (orig idx 1): left eye missing entirely -> falls back to right
        self.assertAlmostEqual(fused[Column.EyeX][1], 150.0)
        self.assertAlmostEqual(fused[Column.EyeY][1], 250.0)
        self.assertAlmostEqual(fused[Column.Diameter][1], 3.5)

        # row 2 (orig idx 3): right eye missing entirely -> falls back to left
        self.assertAlmostEqual(fused[Column.EyeX][2], 300.0)
        self.assertAlmostEqual(fused[Column.EyeY][2], 400.0)
        self.assertAlmostEqual(fused[Column.Diameter][2], 3.0)

    def test_fuse_both_drops_rows_with_only_one_valid_eye(self):
        df = pl.DataFrame(
            {
                Column.Timestamp: [0, 10, 20, 30],
                Column.LeftX: [100.0, 0.0, 0.0, 300.0],
                Column.LeftY: [200.0, 0.0, 0.0, 400.0],
                Column.RightX: [110.0, 150.0, 0.0, 0.0],
                Column.RightY: [210.0, 250.0, 0.0, 0.0],
                Column.LeftDiameter: [3.0, 0.0, 0.0, 3.0],
                Column.RightDiameter: [3.2, 3.5, 0.0, 0.0],
            }
        )

        fused = fuse_raw_data(df, EyeFusionProtocol.Both)

        # Only the row where both eyes are valid is retained.
        self.assertEqual(fused[Column.Timestamp].to_list(), [0])

        self.assertAlmostEqual(fused[Column.EyeX][0], 105.0)
        self.assertAlmostEqual(fused[Column.EyeY][0], 205.0)
        self.assertAlmostEqual(fused[Column.Diameter][0], 3.1)

    def test_fuse_aligned_estimates_missing_eye(self):
        df = pl.DataFrame(
            {
                Column.Timestamp: [0, 10, 20],
                Column.LeftX: [100.0, 0.0, 120.0],
                Column.LeftY: [200.0, 0.0, 220.0],
                Column.RightX: [110.0, 130.0, 130.0],
                Column.RightY: [210.0, 230.0, 230.0],
                Column.LeftDiameter: [3.0, 3.0, 3.0],
                Column.RightDiameter: [3.0, 3.0, 3.0],
            }
        )

        fused = fuse_raw_data(df, EyeFusionProtocol.Aligned)

        # Both eyes have an X/Y offset of -10 in the binocular rows.
        self.assertEqual(fused[Column.Timestamp].to_list(), [0, 10, 20])

        self.assertAlmostEqual(fused[Column.EyeX][0], 105.0)
        self.assertAlmostEqual(fused[Column.EyeX][1], 125.0)
        self.assertAlmostEqual(fused[Column.EyeX][2], 125.0)

        self.assertAlmostEqual(fused[Column.EyeY][0], 205.0)
        self.assertAlmostEqual(fused[Column.EyeY][1], 225.0)
        self.assertAlmostEqual(fused[Column.EyeY][2], 225.0)

        # Diameter should still use ordinary fusion.
        self.assertAlmostEqual(fused[Column.Diameter][0], 3.0)
        self.assertAlmostEqual(fused[Column.Diameter][1], 3.0)
        self.assertAlmostEqual(fused[Column.Diameter][2], 3.0)
