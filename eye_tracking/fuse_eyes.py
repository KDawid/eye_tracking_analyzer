from enum import Enum, auto

import polars as pl

from eye_tracking.columns import Column


class EyeFusionProtocol(Enum):
    # Keep if any eye data is valid; take mean if both are valid.
    Any = auto()

    # Keep only rows where both eyes have valid data; take the mean.
    Both = auto()

    # Take the mean if both eyes are valid. If only one eye is valid, estimate the missing eye using the average
    # left/right offset observed in rows where both are valid.
    Aligned = auto()


def fuse_raw_data(df: pl.DataFrame, protocol=EyeFusionProtocol.Any) -> pl.DataFrame:
    left_valid = _eye_valid(Column.LeftX, Column.LeftY, Column.LeftDiameter)
    right_valid = _eye_valid(Column.RightX, Column.RightY, Column.RightDiameter)

    valid = (
        left_valid & right_valid
        if protocol == EyeFusionProtocol.Both
        else left_valid | right_valid
    )

    if protocol == EyeFusionProtocol.Aligned:
        x_offset = _get_offset(df, Column.LeftX, Column.RightX)
        y_offset = _get_offset(df, Column.LeftY, Column.RightY)
    else:
        x_offset = y_offset = 0.0

    return df.filter(valid).select(
        [
            Column.Timestamp,
            _fuse(Column.LeftX, Column.RightX, Column.EyeX, x_offset),
            _fuse(Column.LeftY, Column.RightY, Column.EyeY, y_offset),
            _fuse(Column.LeftDiameter, Column.RightDiameter, Column.Diameter),
        ]
    )


def _get_offset(df: pl.DataFrame, left_column: str, right_column: str) -> float:
    left, right = pl.col(left_column), pl.col(right_column)
    return df.filter((left > 0) & (right > 0)).select((left - right).mean()).item()


def _eye_valid(x_column: str, y_column: str, diameter_column: str) -> pl.Expr:
    return (
        (pl.col(x_column) > 0) & (pl.col(y_column) > 0) & (pl.col(diameter_column) > 0)
    )


def _fuse(
    left_column: str, right_column: str, target_column: str, offset: float = 0.0
) -> pl.Expr:
    left, right = pl.col(left_column), pl.col(right_column)

    return (
        pl.when((left > 0) & (right > 0))
        .then((left + right) / 2)
        .when(left > 0)
        .then(left - offset / 2)
        .otherwise(right + offset / 2)
        .alias(target_column)
    )
