# TODO github repo: https://github.com/esdalmaijer/PyGazeAnalyser.git
#  install it

import polars as pl
from pygazeanalyser.detectors import fixation_detection

from eye_tracking.columns import Column


def idt(
    df: pl.DataFrame, *, max_distance: float = 25.0, min_duration: float = 100.0
) -> pl.DataFrame:
    data = (
        df.select(Column.EyeX, Column.EyeY, Column.Timestamp)
        .drop_nulls()
        .sort(Column.Timestamp)
    )

    x = data[Column.EyeX].to_numpy()
    y = data[Column.EyeY].to_numpy()
    t = data[Column.Timestamp].to_numpy()

    # This implementation expects milliseconds.
    t_ms = t * 1000.0

    _, end_fixations = fixation_detection(
        x, y, t_ms, missing=None, maxdist=max_distance, mindur=min_duration
    )

    return pl.DataFrame(
        end_fixations,
        schema=["start_timestamp", "end_timestamp", "duration", "eye_x", "eye_y"],
        orient="row",
    ).with_columns(
        (pl.col("start_timestamp") / 1000).alias("start_timestamp"),
        (pl.col("end_timestamp") / 1000).alias("end_timestamp"),
        (pl.col("duration") / 1000).alias("duration"),
    )
