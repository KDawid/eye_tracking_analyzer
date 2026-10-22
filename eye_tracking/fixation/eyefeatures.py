from typing import Literal, TypeAlias, TypedDict, get_args

import polars as pl
from eyefeatures.preprocessing.fixation_extraction import IDT, IHMM, IVT

from eye_tracking.columns import Column
from eye_tracking.fixation._check import (
    check_distance,
    check_distribution_parameters,
    check_gaze_data,
    check_parameter_between,
    check_parameter_greater,
    check_parameter_positive,
)


class _NormalParams(TypedDict):
    loc: float
    scale: float


class _IHMMDistributionParams(TypedDict):
    fixation: _NormalParams
    saccade: _NormalParams


_DistanceType: TypeAlias = Literal["euc", "manhattan", "chebyshev"]
_DistributionType: TypeAlias = Literal["norm"]
_DistributionParamType: TypeAlias = Literal["auto"] | _IHMMDistributionParams


def ivt(
    df: pl.DataFrame, *, velocity_threshold: float = 0.5, min_duration: float = 0.1, distance: _DistanceType = "euc"
) -> pl.DataFrame:
    check_parameter_positive(velocity_threshold, "velocity_threshold")
    check_parameter_positive(min_duration, "min_duration")
    check_distance(distance, get_args(_DistanceType))

    check_gaze_data(df)
    pdf = df.select(Column.EyeX, Column.EyeY, Column.Timestamp).to_pandas()

    detector = IVT(
        x=Column.EyeX,
        y=Column.EyeY,
        t=Column.Timestamp,
        threshold=velocity_threshold,
        min_duration=min_duration,
        distance=distance,
    )

    result = detector.fit_transform(pdf)
    return pl.from_pandas(result)


def idt(
    df: pl.DataFrame,
    *,
    min_duration: float = 0.1,
    max_duration: float = 1.0,
    max_dispersion: float = 0.05,
    distance: _DistanceType = "euc",
) -> pl.DataFrame:
    check_parameter_positive(min_duration, "min_duration")
    check_parameter_greater(max_duration, "max_duration", min_duration, "min_duration")
    check_parameter_positive(max_dispersion, "max_dispersion")
    check_distance(distance, get_args(_DistanceType))

    check_gaze_data(df)
    pdf = df.select(Column.EyeX, Column.EyeY, Column.Timestamp).to_pandas()
    detector = IDT(
        x=Column.EyeX,
        y=Column.EyeY,
        t=Column.Timestamp,
        min_duration=min_duration,
        max_duration=max_duration,
        max_dispersion=max_dispersion,
        distance=distance,
    )
    result = detector.fit_transform(pdf)
    return pl.from_pandas(result)


def ihmm(
    df: pl.DataFrame,
    *,
    fixation_to_saccade_probability: float = 0.05,
    saccade_to_fixation_probability: float = 0.05,
    fixation_distribution: _DistributionType = "norm",
    saccade_distribution: _DistributionType = "norm",
    distribution_parameters: _DistributionParamType = "auto",
) -> pl.DataFrame:
    check_parameter_between(fixation_to_saccade_probability, "fixation_to_saccade_probability", 0.0, 1.0)
    check_parameter_between(saccade_to_fixation_probability, "saccade_to_fixation_probability", 0.0, 1.0)
    check_distribution_parameters(distribution_parameters, "distribution_parameters")

    check_gaze_data(df)
    pdf = df.select(Column.EyeX, Column.EyeY, Column.Timestamp).to_pandas()

    if distribution_parameters == "auto":
        distribution_parameters = _DEFAULT_DISTRIBUTION_PARAMETERS

    detector = IHMM(
        x=Column.EyeX,
        y=Column.EyeY,
        t=Column.Timestamp,
        fix2sac=fixation_to_saccade_probability,
        sac2fix=saccade_to_fixation_probability,
        fix_distrib=fixation_distribution,
        sac_distrib=saccade_distribution,
        distrib_params=distribution_parameters,  # type: ignore[arg-type]
    )
    result = detector.fit_transform(pdf)
    return pl.from_pandas(result)


# see fixation_extraction.py:430-449
_DEFAULT_DISTRIBUTION_PARAMETERS: _IHMMDistributionParams = {
    "fixation": {"loc": 0.002, "scale": 0.001},
    "saccade": {"loc": 0.03, "scale": 0.005},
}
