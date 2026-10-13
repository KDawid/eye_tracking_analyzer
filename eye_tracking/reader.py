from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import polars as pl

from eye_tracking.columns import Column
from eye_tracking.input_formats import TOBII, RecordingFormat

_COORDINATE_COLUMNS = (Column.LeftX, Column.LeftY, Column.RightX, Column.RightY)


def read_raw_data(
    file_path: Path | str, *, fmt: RecordingFormat = TOBII
) -> pl.DataFrame:
    mapping = fmt.columns
    df = pl.read_csv(
        file_path,
        columns=fmt.read_columns,
        schema_overrides={
            source: _SCHEMA[target] for source, target in mapping.items()
        },
    ).rename(mapping)

    # Invalid eye measurements become null, including coordinates and pupil size.
    validity_mapping = fmt.validity.as_dict()
    if validity_mapping:
        df = df.with_columns(
            pl.when(pl.col(validity_col) == 1)
            .then(pl.col(column))
            .otherwise(None)
            .alias(column)
            for eye, validity_col in validity_mapping.items()
            for column in _EYE_COLUMNS[eye]
        ).drop(list(validity_mapping.values()))

    if fmt.normalize:
        df = _normalize_coordinates(df)
        df = _normalize_timestamp(df, fmt.time_unit)

    return df


def read_demo_data(file_path: Path | str) -> pl.DataFrame:
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

    df = pl.DataFrame(
        rows, schema=[Column.EyeX, Column.EyeY, Column.Timestamp], orient="row"
    )
    df = _normalize_coordinates(df, columns=(Column.EyeX, Column.EyeY))
    df = _normalize_timestamp(df, "ms")
    return df


def _normalize_coordinates(
    df: pl.DataFrame, *, columns: Sequence[str] = _COORDINATE_COLUMNS
) -> pl.DataFrame:
    return df.with_columns(
        [
            (
                (pl.col(column) - pl.col(column).min())
                / (pl.col(column).max() - pl.col(column).min())
            ).alias(column)
            for column in columns
        ]
    )


def _normalize_timestamp(
    df: pl.DataFrame, time_unit: Literal["s", "ms", "us", "ns"]
) -> pl.DataFrame:
    return df.sort(Column.Timestamp).with_columns(
        (
            (pl.col(Column.Timestamp) - pl.col(Column.Timestamp).min()).cast(pl.Float64)
            * _TO_SECONDS[time_unit]
        ).alias(Column.Timestamp)
    )


_EYE_COLUMNS = {
    "left": (
        Column.LeftX,
        Column.LeftY,
        Column.LeftDiameter,
    ),
    "right": (
        Column.RightX,
        Column.RightY,
        Column.RightDiameter,
    ),
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

_TO_SECONDS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
