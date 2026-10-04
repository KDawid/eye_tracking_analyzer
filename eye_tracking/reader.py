from pathlib import Path

import polars as pl

from eye_tracking.columns import Column


def read_raw_data(
    file_path: Path,
    *,
    column_mapping: dict[str, str] | None = None,
    normalize: bool = False,
) -> pl.DataFrame:
    mapping = column_mapping or {column: column for column in _SCHEMA}

    df = pl.read_csv(
        file_path,
        columns=list(mapping),
        schema_overrides={
            source: _SCHEMA[target] for source, target in mapping.items()
        },
    ).rename(mapping)

    if normalize:
        df = _normalize(df)
    return df


def _normalize(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (
                (pl.col(column) - pl.col(column).filter(pl.col(column) > 0).min())
                / (
                    pl.col(column).filter(pl.col(column) > 0).max()
                    - pl.col(column).filter(pl.col(column) > 0).min()
                )
            ).alias(column)
            for column in _COORDINATE_COLUMNS
        ]
    )


TOBII_COLUMNS = {
    "Time": Column.Timestamp,
    "L Raw X [px]": Column.LeftX,
    "L Raw Y [px]": Column.LeftY,
    "R Raw X [px]": Column.RightX,
    "R Raw Y [px]": Column.RightY,
    "L Mapped Diameter [mm]": Column.LeftDiameter,
    "R Mapped Diameter [mm]": Column.RightDiameter,
}

_COORDINATE_COLUMNS = (Column.LeftX, Column.LeftY, Column.RightX, Column.RightY)
_SCHEMA = {
    Column.Timestamp: pl.UInt64,
    Column.LeftX: pl.Float64,
    Column.LeftY: pl.Float64,
    Column.RightX: pl.Float64,
    Column.RightY: pl.Float64,
    Column.LeftDiameter: pl.Float64,
    Column.RightDiameter: pl.Float64,
}
