from collections.abc import Sequence
from pathlib import Path

import polars as pl

from eye_tracking.columns import Column
from eye_tracking.input_formats import TOBII, CsvFormat, MatlabFormat, TimeUnit

_COORDINATE_COLUMNS = (Column.LeftX, Column.LeftY, Column.RightX, Column.RightY)


def read_raw_data(file_path: Path, *, fmt: CsvFormat = TOBII) -> pl.DataFrame:
    mapping = fmt.columns
    df = pl.read_csv(
        file_path,
        columns=fmt.read_columns,
        schema_overrides={source: _SCHEMA[target] for source, target in mapping.items()},
    ).rename(mapping)

    # Invalid eye measurements become null, including coordinates and pupil size.
    validity_mapping = fmt.validity.as_dict()
    if validity_mapping:
        df = df.with_columns(
            pl.when(pl.col(validity_col) == 1).then(pl.col(column)).otherwise(None).alias(column)
            for eye, validity_col in validity_mapping.items()
            for column in _EYE_COLUMNS[eye]
        ).drop(list(validity_mapping.values()))

    df = _normalize_coordinates(df)
    df = _timestamp_to_ms(df, fmt.time_unit, zero_based=fmt.normalize)
    return df


def read_demo_data(file_path: Path) -> pl.DataFrame:
    rows, errors = [], []

    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) == 4 and fields[:2] == ["0.099375", "0"]:
                fields.pop(1)
            try:
                if len(fields) != 3:
                    raise ValueError
                rows.append(tuple(map(float, fields)))
            except ValueError:
                errors.append(i)

    if errors:
        raise ValueError(f"Malformed rows: {errors[:10]} ({len(errors)} total)")

    df = pl.DataFrame(rows, schema=[Column.EyeX, Column.EyeY, Column.Timestamp], orient="row")
    df = _normalize_coordinates(df, columns=(Column.EyeX, Column.EyeY))
    df = _timestamp_to_ms(df, "ms")
    return df


def read_matlab_data(file_path: Path, fmt: MatlabFormat, *, drop_missing: bool = True) -> pl.DataFrame:
    from scipy.io import loadmat  # only needed for .mat files

    matrix = loadmat(file_path, simplify_cells=True)[fmt.variable][fmt.field_name]
    df = pl.DataFrame(matrix, schema=list(fmt.columns), orient="row")

    df = df.with_columns(
        pl.when(pl.col(Column.Validity) == 1).then(pl.col(column)).otherwise(None).alias(column)
        for column in (Column.EyeX, Column.EyeY, Column.LeftDiameter, Column.RightDiameter)
    ).select(
        Column.Timestamp,
        Column.EyeX,
        Column.EyeY,
        pl.mean_horizontal(Column.LeftDiameter, Column.RightDiameter).alias(Column.Diameter),
    )

    if drop_missing:  # same default as fuse_raw_data
        df = df.drop_nulls([Column.EyeX, Column.EyeY])

    df = _normalize_coordinates(df, columns=(Column.EyeX, Column.EyeY))
    return _timestamp_to_ms(df, fmt.time_unit, zero_based=fmt.normalize)


def _normalize_coordinates(df: pl.DataFrame, *, columns: Sequence[str] = _COORDINATE_COLUMNS) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.when(pl.col(column).max() > pl.col(column).min())
            .then((pl.col(column) - pl.col(column).min()) / (pl.col(column).max() - pl.col(column).min()))
            .otherwise(pl.col(column) * 0.0)
            .alias(column)
            for column in columns
        ]
    )


def _timestamp_to_ms(df: pl.DataFrame, time_unit: TimeUnit, *, zero_based: bool = True) -> pl.DataFrame:
    timestamp = pl.col(Column.Timestamp).cast(pl.Float64)
    if zero_based:
        timestamp = timestamp - timestamp.min()
    return df.sort(Column.Timestamp).with_columns((timestamp * _TO_MS[time_unit]).alias(Column.Timestamp))


_EYE_COLUMNS = {
    "left": (Column.LeftX, Column.LeftY, Column.LeftDiameter),
    "right": (Column.RightX, Column.RightY, Column.RightDiameter),
}

_SCHEMA = {
    Column.Timestamp: pl.Float64,
    Column.LeftX: pl.Float64,
    Column.LeftY: pl.Float64,
    Column.RightX: pl.Float64,
    Column.RightY: pl.Float64,
    Column.LeftDiameter: pl.Float64,
    Column.RightDiameter: pl.Float64,
}

_TO_MS = {"s": 1e3, "ms": 1.0, "us": 1e-3, "ns": 1e-6}
