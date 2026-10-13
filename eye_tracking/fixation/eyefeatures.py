import polars as pl
from eyefeatures.preprocessing.fixation_extraction import IDT, IHMM, IVT

from eye_tracking.columns import Column


def ivt(
    df: pl.DataFrame,
    *,
    velocity_threshold: float = 0.5,
    min_duration: float = 0.1,
) -> pl.DataFrame:
    pdf = df.select(Column.EyeX, Column.EyeY, Column.Timestamp).to_pandas()

    detector = IVT(
        x=Column.EyeX,
        y=Column.EyeY,
        t=Column.Timestamp,
        threshold=velocity_threshold,
        min_duration=min_duration,
    )

    result = detector.fit_transform(pdf)
    return pl.from_pandas(result)


def idt(
    df: pl.DataFrame,
    *,
    min_duration: float = 0.1,
    max_duration: float = 1.0,
    max_dispersion: float = 0.05,
) -> pl.DataFrame:
    pdf = df.select(Column.EyeX, Column.EyeY, Column.Timestamp).to_pandas()
    detector = IDT(
        x=Column.EyeX,
        y=Column.EyeY,
        t=Column.Timestamp,
        min_duration=min_duration,
        max_duration=max_duration,
        max_dispersion=max_dispersion,
        distance="euc",
    )
    result = detector.fit_transform(pdf)
    return pl.from_pandas(result)


def ihmm(df: pl.DataFrame) -> pl.DataFrame:
    pdf = df.select(Column.EyeX, Column.EyeY, Column.Timestamp).to_pandas()

    detector = IHMM(x=Column.EyeX, y=Column.EyeY, t=Column.Timestamp)
    result = detector.fit_transform(pdf)
    return pl.from_pandas(result)
