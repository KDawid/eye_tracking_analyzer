import logging
from enum import Enum, auto

import polars as pl

from eye_tracking.columns import Column


class EyeFusionProtocol(Enum):
    Left = "left"
    Right = "right"

    # Keep if any eye data is valid; take mean if both are valid.
    Any = "any"

    # Keep only rows where both eyes have valid data; take the mean.
    Both = "both"

    # Take the mean if both eyes are valid. If only one eye is valid, estimate the missing eye using the average
    # left/right offset observed in rows where both are valid.
    Aligned = "aligned"

    @property
    def label(self) -> str:
        labels = {
            EyeFusionProtocol.Any: "Any (mean when both available)",
            EyeFusionProtocol.Both: "Both (require both eyes)",
            EyeFusionProtocol.Aligned: "Aligned (compensate for eye offset)",
            EyeFusionProtocol.Left: "Left eye",
            EyeFusionProtocol.Right: "Right eye",
        }
        return labels[self]


def fuse_raw_data(
    df: pl.DataFrame, protocol: EyeFusionProtocol = EyeFusionProtocol.Any, *, drop_missing: bool = True
) -> pl.DataFrame:
    left_valid = _eye_valid(Column.LeftX, Column.LeftY) & pl.lit(protocol != EyeFusionProtocol.Right)
    right_valid = _eye_valid(Column.RightX, Column.RightY) & pl.lit(protocol != EyeFusionProtocol.Left)

    allow_single_eye = protocol != EyeFusionProtocol.Both
    offset_x = offset_y = 0.0
    if protocol == EyeFusionProtocol.Aligned:
        both_valid = left_valid & right_valid
        offset_x = _get_offset(df, Column.LeftX, Column.RightX, both_valid)
        offset_y = _get_offset(df, Column.LeftY, Column.RightY, both_valid)

    fused = df.select(
        Column.Timestamp,
        _fuse(Column.LeftX, Column.RightX, left_valid, right_valid, allow_single_eye, offset_x).alias(Column.EyeX),
        _fuse(Column.LeftY, Column.RightY, left_valid, right_valid, allow_single_eye, offset_y).alias(Column.EyeY),
        _mean_diameter(protocol).alias(Column.Diameter),
    )
    return fused.drop_nulls([Column.EyeX, Column.EyeY]) if drop_missing else fused


def _get_offset(df: pl.DataFrame, left_column: str, right_column: str, both_valid: pl.Expr) -> float:
    offset = df.filter(both_valid).select((pl.col(left_column) - pl.col(right_column)).mean()).item()
    if offset is None:
        logging.warning("No binocular samples found!")
        return 0.0
    return offset  # TODO: warn when there are no binocular samples


def _eye_valid(x_column: str, y_column: str) -> pl.Expr:
    return pl.col(x_column).is_not_null() & pl.col(y_column).is_not_null()


def _fuse(
    left_col: str, right_col: str, left_valid: pl.Expr, right_valid: pl.Expr, allow_single_eye: bool, offset: float
) -> pl.Expr:
    left, right = pl.col(left_col), pl.col(right_col)
    fused = pl.when(left_valid & right_valid).then((left + right) / 2)
    if allow_single_eye:
        fused = fused.when(left_valid).then(left - offset / 2)
        fused = fused.when(right_valid).then(right + offset / 2)
    return fused.otherwise(None)


def _mean_diameter(protocol: EyeFusionProtocol) -> pl.Expr:
    if protocol == EyeFusionProtocol.Left:
        return pl.col(Column.LeftDiameter)
    if protocol == EyeFusionProtocol.Right:
        return pl.col(Column.RightDiameter)
    return pl.mean_horizontal(pl.col(Column.LeftDiameter), pl.col(Column.RightDiameter))
