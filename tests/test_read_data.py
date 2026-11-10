"""
Unit tests for the recording readers (eye_tracking.input_formats and eye_tracking.reader).

Run from the repository root:   python -m unittest tests.test_reader -v     (pytest also works)

Contract under test, for every reader:
  * timestamps are milliseconds; zero-based when the format says `normalize`,
  * coordinates are min-max normalised to [0, 1] per column, ignoring missing samples,
  * invalid samples are null, never silently kept or turned into NaN.

The two private helpers hold the logic all readers share, so they are tested directly with tiny frames; the readers
themselves are only tested for what they add (file layout, validity masking, wiring of the format settings).
"""

from __future__ import annotations

import importlib.util
import re
import tempfile
import unittest
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from eye_tracking.columns import Column
from eye_tracking.input_formats import (
    MATLAB,
    SIMPLEGAZETRACKER,
    TOBII,
    CsvFormat,
    MatlabFormat,
    TimeUnit,
    ValidityMapping,
)
from eye_tracking.reader import (
    _SCHEMA,
    _normalize_coordinates,
    _timestamp_to_ms,
    read_demo_data,
    read_matlab_data,
    read_raw_data,
)

LEFT_EYE = (Column.LeftX, Column.LeftY, Column.LeftDiameter)
RIGHT_EYE = (Column.RightX, Column.RightY, Column.RightDiameter)
SAMPLE_COUNT = 4

# One binocular recording of four samples, 8 ms apart if the unit is microseconds.
DEFAULT_SAMPLES = {
    Column.Timestamp: [5_000_000, 5_008_000, 5_016_000, 5_024_000],
    Column.LeftX: [100.0, 200.0, 300.0, 400.0],
    Column.LeftY: [10.0, 20.0, 30.0, 40.0],
    Column.RightX: [1000.0, 1100.0, 1200.0, 1300.0],
    Column.RightY: [60.0, 70.0, 80.0, 90.0],
    Column.LeftDiameter: [3.0] * SAMPLE_COUNT,
    Column.RightDiameter: [4.0] * SAMPLE_COUNT,
}


LOST = 0.0
FIXATION, BLINK = 1, 5


def csv_recording(
    fmt: CsvFormat = TOBII, *, samples: dict | None = None, left_validity: list[int] | None = None
) -> pl.DataFrame:
    """The recording above in the file layout of `fmt`; every sample is valid unless `left_validity` says otherwise."""
    values = {**DEFAULT_SAMPLES, **(samples or {})}
    columns = {source: values[target] for source, target in fmt.columns.items()}
    for validity_column in fmt.validity.as_dict().values():
        columns[validity_column] = [1] * SAMPLE_COUNT
    if left_validity is not None:
        columns[fmt.validity.left] = left_validity
    return pl.DataFrame(columns)


class FrameTestCase(unittest.TestCase):
    def assertColumn(self, frame: pl.DataFrame, column: str, expected: list[float | None]) -> None:
        """Element-wise comparison; None must be null (NaN does not count), floats must match closely."""
        actual = frame[column].to_list()
        self.assertEqual(len(actual), len(expected), f"{column}: wrong number of rows")
        for index, (got, want) in enumerate(zip(actual, expected)):
            if want is None:
                self.assertIsNone(got, f"{column}[{index}] should be null")
            else:
                self.assertIsNotNone(got, f"{column}[{index}] should not be null")
                self.assertAlmostEqual(got, want, places=9, msg=f"{column}[{index}]")


class FileTestCase(FrameTestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def write_csv(self, frame: pl.DataFrame) -> Path:
        path = self.directory / "recording.csv"
        frame.write_csv(path)
        return path

    def write_text(self, text: str) -> Path:
        path = self.directory / "demo.dat"
        path.write_text(text, encoding="utf-8")
        return path


class FormatTests(unittest.TestCase):
    def test_validity_mapping_lists_only_configured_eyes(self):
        self.assertEqual(ValidityMapping(left="LV").as_dict(), {"left": "LV"})
        self.assertEqual(ValidityMapping().as_dict(), {})

    def test_read_columns_list_mapped_columns_then_validity_columns_without_duplicates(self):
        fmt = CsvFormat(
            columns={"Time": Column.Timestamp, "LV": Column.LeftX}, validity=ValidityMapping(left="LV", right="RV")
        )
        self.assertEqual(fmt.read_columns, ["Time", "LV", "RV"])

    def test_preset_formats_only_use_columns_the_readers_know(self):
        for fmt in (TOBII, SIMPLEGAZETRACKER):
            with self.subTest(fmt=fmt):
                self.assertLessEqual(set(fmt.columns.values()), set(_SCHEMA))
        required_by_matlab_reader = {
            Column.Timestamp,
            Column.EyeX,
            Column.EyeY,
            Column.LeftDiameter,
            Column.RightDiameter,
            Column.Label,
        }
        self.assertLessEqual(required_by_matlab_reader, set(MATLAB.columns))


class NormalizeCoordinatesTests(FrameTestCase):
    def test_scales_a_column_to_the_unit_range_ignoring_nulls(self):
        frame = pl.DataFrame({Column.EyeX: [2.0, 4.0, None, 6.0]})

        result = _normalize_coordinates(frame, columns=(Column.EyeX,))

        self.assertColumn(result, Column.EyeX, [0.0, 0.5, None, 1.0])

    def test_scales_each_requested_column_independently_and_leaves_the_rest_alone(self):
        frame = pl.DataFrame({Column.EyeX: [0.0, 10.0], Column.EyeY: [100.0, 300.0], Column.Diameter: [3.0, 5.0]})

        result = _normalize_coordinates(frame, columns=(Column.EyeX, Column.EyeY))

        self.assertColumn(result, Column.EyeX, [0.0, 1.0])
        self.assertColumn(result, Column.EyeY, [0.0, 1.0])
        self.assertColumn(result, Column.Diameter, [3.0, 5.0])

    def test_constant_and_all_null_columns_never_become_nan(self):
        cases = {"constant": ([5.0, 5.0, None, 5.0], [0.0, 0.0, None, 0.0]), "all null": ([None, None], [None, None])}
        for name, (values, expected) in cases.items():
            with self.subTest(name):
                frame = pl.DataFrame({Column.EyeX: pl.Series(values, dtype=pl.Float64)})

                result = _normalize_coordinates(frame, columns=(Column.EyeX,))

                self.assertColumn(result, Column.EyeX, expected)


class TimestampToMsTests(FrameTestCase):
    TICKS_PER_MS: dict[TimeUnit, float] = {"s": 0.001, "ms": 1.0, "us": 1_000.0, "ns": 1_000_000.0}

    def test_converts_every_unit_to_zero_based_milliseconds(self):
        for unit, ticks_per_ms in self.TICKS_PER_MS.items():
            with self.subTest(unit=unit):
                frame = pl.DataFrame({Column.Timestamp: [100.0 + k * ticks_per_ms for k in (0, 1, 3)]})

                result = _timestamp_to_ms(frame, unit)

                self.assertColumn(result, Column.Timestamp, [0.0, 1.0, 3.0])

    def test_keeps_the_original_origin_when_not_zero_based(self):
        frame = pl.DataFrame({Column.Timestamp: [5_000.0, 6_000.0]})

        result = _timestamp_to_ms(frame, "us", zero_based=False)

        self.assertColumn(result, Column.Timestamp, [5.0, 6.0])

    def test_sorts_rows_by_time_and_keeps_other_columns_aligned(self):
        frame = pl.DataFrame({Column.Timestamp: [30.0, 10.0, 20.0], "marker": ["c", "a", "b"]})

        result = _timestamp_to_ms(frame, "ms")

        self.assertColumn(result, Column.Timestamp, [0.0, 10.0, 20.0])
        self.assertEqual(result["marker"].to_list(), ["a", "b", "c"])


class ReadRawDataTests(FileTestCase):
    def test_renames_file_columns_drops_validity_columns_and_ignores_unmapped_ones(self):
        frame = csv_recording().with_columns(pl.lit("fixation").alias("Event"))

        result = read_raw_data(self.write_csv(frame))

        self.assertCountEqual(result.columns, TOBII.columns.values())

    def test_invalid_samples_are_nulled_per_eye_including_the_pupil(self):
        frame = csv_recording(left_validity=[1, 0, 4, 1])  # anything but 1 is invalid

        result = read_raw_data(self.write_csv(frame))

        for column in LEFT_EYE:
            self.assertEqual(result[column].is_null().to_list(), [False, True, True, False], column)
        for column in RIGHT_EYE:
            self.assertEqual(result[column].null_count(), 0, column)

    def test_format_without_validity_mapping_keeps_every_sample(self):
        frame = csv_recording(SIMPLEGAZETRACKER)

        result = read_raw_data(self.write_csv(frame), fmt=SIMPLEGAZETRACKER)

        self.assertEqual(result.height, SAMPLE_COUNT)
        self.assertEqual(sum(result.null_count().row(0)), 0)

    def test_timestamps_follow_the_time_unit_and_origin_setting_of_the_format(self):
        path = self.write_csv(csv_recording())
        cases = {
            "zero based": (TOBII, [0.0, 8.0, 16.0, 24.0]),
            "original origin": (replace(TOBII, normalize=False), [5_000.0, 5_008.0, 5_016.0, 5_024.0]),
        }
        for name, (fmt, expected) in cases.items():
            with self.subTest(name):
                self.assertColumn(read_raw_data(path, fmt=fmt), Column.Timestamp, expected)

    def test_invalid_samples_are_masked_before_the_coordinates_are_normalised(self):
        frame = csv_recording(
            samples={Column.LeftX: [100.0, 999.0, 300.0, 500.0]},  # 999 is an invalid outlier
            left_validity=[1, 0, 1, 1],
        )

        result = read_raw_data(self.write_csv(frame))

        self.assertColumn(result, Column.LeftX, [0.0, None, 0.5, 1.0])
        self.assertColumn(result, Column.RightX, [0.0, 1 / 3, 2 / 3, 1.0])  # the other eye has its own scale


class ReadDemoDataTests(FileTestCase):
    def test_parses_whitespace_separated_rows_skipping_blank_lines(self):
        path = self.write_text("0.10 0.20 100\n\n0.20 0.40 110\n0.30 0.60 120\n")

        result = read_demo_data(path)

        self.assertEqual(result.columns, [Column.EyeX, Column.EyeY, Column.Timestamp])
        self.assertColumn(result, Column.EyeX, [0.0, 0.5, 1.0])
        self.assertColumn(result, Column.EyeY, [0.0, 0.5, 1.0])
        self.assertColumn(result, Column.Timestamp, [0.0, 10.0, 20.0])

    def test_repairs_the_known_four_field_row(self):
        # "0.099375 0 <y> <t>": the stray "0" is dropped, so x=0.099375 (the minimum) and y=0.50 (the maximum)
        path = self.write_text("0.10 0.20 100\n0.099375 0 0.50 110\n0.30 0.40 120\n")

        result = read_demo_data(path)

        self.assertColumn(result, Column.EyeX, [result[Column.EyeX][0], 0.0, 1.0])
        self.assertColumn(result, Column.EyeY, [0.0, 1.0, result[Column.EyeY][2]])
        self.assertColumn(result, Column.Timestamp, [0.0, 10.0, 20.0])

    def test_reports_malformed_lines_by_file_line_number(self):
        # line 2 is blank and still counts; line 3 has too few fields, line 4 is not numeric
        path = self.write_text("0.1 0.2 100\n\n0.3 0.4\nx y z\n0.5 0.6 120\n")

        with self.assertRaisesRegex(ValueError, re.escape("Malformed rows: [3, 4] (2 total)")):
            read_demo_data(path)

    def test_lists_at_most_ten_malformed_lines_but_counts_all(self):
        path = self.write_text("bad\n" * 12)

        with self.assertRaisesRegex(ValueError, re.escape(f"Malformed rows: {list(range(1, 11))} (12 total)")):
            read_demo_data(path)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is not installed")
class ReadMatlabDataTests(FileTestCase):
    # [timestamp (us), left pupil, right pupil, x, y, label]; the tracker loses the eye on the second sample
    ROWS = (
        [5_000_000, 3.0, 3.2, 100.0, 10.0, FIXATION],
        [5_008_000, LOST, LOST, LOST, LOST, BLINK],
        [5_016_000, 3.0, 3.2, 300.0, 30.0, FIXATION],
        [5_024_000, 3.0, 3.2, 500.0, 50.0, FIXATION],
    )

    def write_mat(self, rows: Sequence[list[float]], *, variable: str = "ETdata", field: str = "pos") -> Path:
        from scipy.io import savemat

        path = self.directory / "gaze.mat"
        savemat(path, {variable: {field: np.array(rows, dtype=float)}})
        return path

    def test_returns_the_fused_table_with_missing_samples_as_null_and_ignored_by_the_scale(self):
        result = read_matlab_data(self.write_mat(self.ROWS), MATLAB, drop_missing=False)

        self.assertEqual(result.columns, [Column.Timestamp, Column.EyeX, Column.EyeY, Column.Diameter, Column.Label])
        self.assertColumn(result, Column.Timestamp, [0.0, 8.0, 16.0, 24.0])
        self.assertColumn(result, Column.EyeX, [0.0, None, 0.5, 1.0])
        self.assertColumn(result, Column.EyeY, [0.0, None, 0.5, 1.0])
        self.assertColumn(result, Column.Diameter, [3.1, None, 3.1, 3.1])  # mean of both pupils

    def test_drops_missing_samples_by_default(self):
        result = read_matlab_data(self.write_mat(self.ROWS), MATLAB)

        self.assertColumn(result, Column.Timestamp, [0.0, 16.0, 24.0])

    def test_normalize_setting_controls_the_time_origin(self):
        result = read_matlab_data(self.write_mat(self.ROWS), replace(MATLAB, normalize=False))

        self.assertColumn(result, Column.Timestamp, [5_000.0, 5_016.0, 5_024.0])

    def test_uses_the_column_order_variable_and_field_names_of_the_format(self):
        fmt = MatlabFormat(
            columns=(
                Column.Label,
                Column.Timestamp,
                Column.EyeX,
                Column.EyeY,
                Column.LeftDiameter,
                Column.RightDiameter,
            ),
            variable="Recording",
            field_name="samples",
        )
        rows = [[1, 1_000_000, 10.0, 70.0, 2.0, 4.0], [1, 1_002_000, 30.0, 90.0, 2.0, 4.0]]

        result = read_matlab_data(self.write_mat(rows, variable="Recording", field="samples"), fmt)

        self.assertColumn(result, Column.Timestamp, [0.0, 2.0])
        self.assertColumn(result, Column.EyeX, [0.0, 1.0])
        self.assertColumn(result, Column.EyeY, [0.0, 1.0])
        self.assertColumn(result, Column.Diameter, [3.0, 3.0])

    def test_time_origin_is_the_first_recorded_sample_even_when_it_is_lost(self):
        rows = [
            [5_000_000, LOST, LOST, LOST, LOST, BLINK],
            [5_008_000, 3.0, 3.2, 200.0, 20.0, FIXATION],
            [5_016_000, 3.0, 3.2, 300.0, 30.0, FIXATION],
        ]

        result = read_matlab_data(self.write_mat(rows), MATLAB)

        self.assertColumn(result, Column.Timestamp, [8.0, 16.0])

    def test_lost_samples_become_null_and_do_not_stretch_the_scale(self):
        result = read_matlab_data(self.write_mat(self.ROWS), MATLAB, drop_missing=False)

        self.assertEqual(result.columns, [Column.Timestamp, Column.EyeX, Column.EyeY, Column.Diameter, Column.Label])
        self.assertColumn(result, Column.Timestamp, [0.0, 8.0, 16.0, 24.0])
        self.assertColumn(result, Column.EyeX, [0.0, None, 0.5, 1.0])
        self.assertColumn(result, Column.EyeY, [0.0, None, 0.5, 1.0])
        self.assertColumn(result, Column.Diameter, [3.1, None, 3.1, 3.1])

    def test_validity_comes_from_the_data_not_from_the_label(self):
        rows = [
            [5_000_000, 3.0, 3.2, 100.0, 10.0, BLINK],  # annotated as blink, but the tracker still reports gaze
            [5_008_000, LOST, LOST, LOST, LOST, FIXATION],  # annotated as fixation, but the tracker lost the eye
            [5_016_000, 3.0, 3.2, 300.0, 30.0, FIXATION],
        ]

        result = read_matlab_data(self.write_mat(rows), MATLAB, drop_missing=False)

        self.assertColumn(result, Column.EyeX, [0.0, None, 1.0])
        self.assertColumn(result, Column.Label, [BLINK, FIXATION, FIXATION])  # carried through unchanged

    def test_format_without_a_lost_sample_value_keeps_zeros_as_data(self):
        result = read_matlab_data(self.write_mat(self.ROWS), replace(MATLAB, lost_sample_value=None))

        self.assertEqual(result.height, len(self.ROWS))
        self.assertEqual(result[Column.EyeX].null_count(), 0)


if __name__ == "__main__":
    unittest.main()
