import math
from collections.abc import Collection
from numbers import Real

import numpy as np
import polars as pl

from eye_tracking.columns import Column


def check_gaze_data(df: pl.DataFrame):
    if not isinstance(df, pl.DataFrame):
        raise TypeError("'df' must be a Polars DataFrame.")

    required = [Column.EyeX, Column.EyeY, Column.Timestamp]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.height < 2:
        raise ValueError("At least two gaze samples are required.")

    if any(df[col].null_count() > 0 for col in required):
        raise ValueError("Gaze coordinates and timestamps must not contain nulls.")

    pdf = df.select(required).to_pandas()
    values = pdf.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Gaze coordinates and timestamps must be finite.")

    timestamps = pdf[Column.Timestamp].to_numpy(dtype=float)
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("Timestamps must be strictly increasing.")


def check_parameter_positive(parameter: float, name: str):
    _check_finite(parameter, name)
    if parameter <= 0:
        raise ValueError(f"Parameter '{name}' must be positive.")


def check_distance(distance: str, values: Collection[str]):
    if distance not in values:
        raise ValueError(f"'distance' must be one of {tuple(values)}, got '{distance}'.")


def check_parameter_greater(parameter1: float, name1: str, parameter2: float, name2: str):
    _check_finite(parameter1, name1)
    _check_finite(parameter2, name2)
    if parameter1 <= parameter2:
        raise ValueError(f"Parameter '{name1}' must be greater than '{name2}'.")


def check_parameter_between(parameter: float, name: str, lower: float, upper: float):
    _check_finite(parameter, name)
    if not lower < parameter < upper:
        raise ValueError(f"Parameter '{name}' must be between {lower} and {upper}.")


def check_distribution_parameters(params: object, name: str):
    if isinstance(params, str):
        if params == "auto":
            return
        raise ValueError(f"'{name}' must be 'auto' or a dictionary.")

    if not isinstance(params, dict):
        raise TypeError(f"'{name}' must be 'auto' or a dictionary.")

    if set(params) != {"fixation", "saccade"}:
        raise ValueError(f"'{name}' must contain exactly 'fixation' and 'saccade'.")

    for state in ("fixation", "saccade"):
        state_params = params[state]

        if not isinstance(state_params, dict):
            raise TypeError(f"Parameters for '{state}' must be a dictionary.")

        if set(state_params) != {"loc", "scale"}:
            raise ValueError(f"Parameters for '{state}' must contain exactly 'loc' and 'scale'.")

        _check_finite(state_params["loc"], f"{state}.loc")
        check_parameter_positive(state_params["scale"], f"{state}.scale")


def _check_finite(parameter: Real, name: str):
    if isinstance(parameter, bool) or not isinstance(parameter, Real):
        raise TypeError(f"Parameter '{name}' must be a real number.")

    if not math.isfinite(parameter):
        raise ValueError(f"Parameter '{name}' must be finite.")
