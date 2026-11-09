import logging

import plotly.graph_objects as go
import polars as pl

from eye_tracking.columns import Column


def fused_gaze_figure(fused: pl.DataFrame) -> go.Figure:
    fig = go.Figure()

    for column, name, color in (
        (Column.EyeX, "Fused X", _get_color("fused_x")),
        (Column.EyeY, "Fused Y", _get_color("fused_y")),
    ):
        fig.add_trace(
            go.Scattergl(
                x=fused[Column.Timestamp].to_list(),
                y=fused[column].to_list(),
                mode="lines",
                name=name,
                line={"color": color},
            )
        )

    fig.update_layout(
        title="Fused gaze data",
        xaxis_title="Time (ms)",
        yaxis_title="Normalized coordinate",
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def fixation_figure(gaze: pl.DataFrame, fixations: pl.DataFrame, *, algorithm: str, parameters: dict) -> go.Figure:
    fig = go.Figure()

    # Raw gaze trajectory
    fig.add_trace(
        go.Scattergl(
            x=gaze[Column.EyeX].to_list(),
            y=gaze[Column.EyeY].to_list(),
            mode="lines",
            name="Gaze trajectory",
            line={"color": _get_color("gaze_trajectory"), "width": 1},
        )
    )

    # Fixation centers
    if fixations.height:
        fig.add_trace(
            go.Scattergl(
                x=fixations[Column.EyeX].to_list(),
                y=fixations[Column.EyeY].to_list(),
                mode="markers",
                name="Fixations",
                marker={"color": _get_color("fixations"), "size": 10, "symbol": "circle"},
                customdata=list(
                    zip(
                        fixations[Column.StartTimestamp].to_list(),
                        fixations[Column.EndTimestamp].to_list(),
                        fixations[Column.Duration].to_list(),
                    )
                ),
                hovertemplate=(
                    "Start: %{customdata[0]:.2f} ms<br>"
                    "End: %{customdata[1]:.2f} ms<br>"
                    "Duration: %{customdata[2]:.2f} ms"
                    "<extra></extra>"
                ),
            )
        )

    count, total_duration, average_duration = _get_fixation_statistics(fixations)

    fig.update_layout(
        title={
            "text": (
                f"{algorithm} | "
                f"Fixations: {count} | "
                f"Total duration: {total_duration:.1f} ms | "
                f"Average: {average_duration:.1f} ms"
            )
        },
        xaxis={"title": "Normalized X coordinate", "range": [0, 1], "constrain": "domain"},
        yaxis={
            "title": "Normalized Y coordinate",
            "range": [0, 1],
            "scaleanchor": "x",
            "scaleratio": 1,
            "constrain": "domain",
        },
        width=800,
        height=800,
        hovermode="closest",
        template="plotly_white",
    )

    return fig


def gaze_x_figure(raw: pl.DataFrame, fused_data: dict[str, pl.DataFrame], selected_method: str) -> go.Figure:
    fig = go.Figure()

    _add_time_series(fig, raw, Column.LeftX, "Left eye X", _get_color("left_eye"))
    _add_time_series(fig, raw, Column.RightX, "Right eye X", _get_color("right_eye"))

    for method, fused in fused_data.items():
        _add_time_series(
            fig,
            fused,
            Column.EyeX,
            f"Fused X ({method})",
            _get_color(method),
            visible=True if method == selected_method else "legendonly",
        )

    fig.update_layout(
        title="Horizontal gaze coordinate",
        xaxis_title="Time (ms)",
        yaxis_title="Normalized X coordinate",
        yaxis={"range": [0, 1]},
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def gaze_y_figure(raw: pl.DataFrame, fused_data: dict[str, pl.DataFrame], selected_method: str) -> go.Figure:
    fig = go.Figure()

    _add_time_series(fig, raw, Column.LeftY, "Left eye Y", _get_color("left_eye"))
    _add_time_series(fig, raw, Column.RightY, "Right eye Y", _get_color("right_eye"))

    for method, fused in fused_data.items():
        _add_time_series(
            fig,
            fused,
            Column.EyeY,
            f"Fused Y ({method})",
            _get_color(method),
            visible=True if method == selected_method else "legendonly",
        )

    fig.update_layout(
        title="Vertical gaze coordinate",
        xaxis_title="Time (ms)",
        yaxis_title="Normalized Y coordinate",
        yaxis={"range": [0, 1]},
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def pupil_diameter_figure(raw: pl.DataFrame, fused_data: dict[str, pl.DataFrame], selected_method: str) -> go.Figure:
    fig = go.Figure()

    _add_time_series(fig, raw, Column.LeftDiameter, "Left eye diameter", _get_color("left_eye"))
    _add_time_series(fig, raw, Column.RightDiameter, "Right eye diameter", _get_color("right_eye"))

    for method, fused in fused_data.items():
        _add_time_series(
            fig,
            fused,
            Column.Diameter,
            f"Fused diameter ({method})",
            _get_color(method),
            visible=True if method == selected_method else "legendonly",
        )

    fig.update_layout(
        title="Pupil diameter",
        xaxis_title="Time (ms)",
        yaxis_title="Pupil diameter",
        hovermode="x unified",
        template="plotly_white",
    )

    if not fig.data:
        fig.add_annotation(
            text="Pupil diameter data is not available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )

    return fig


def gaze_trajectory_figure(fused: pl.DataFrame, method: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=fused[Column.EyeX].to_list(),
            y=fused[Column.EyeY].to_list(),
            mode="lines",
            name=f"Gaze trajectory ({method})",
            line={"color": _get_color("gaze_trajectory"), "width": 1},
            connectgaps=False,
        )
    )

    fig.update_layout(
        title=f"2D gaze trajectory ({method})",
        xaxis={"title": "Normalized X coordinate", "range": [0, 1], "constrain": "domain"},
        yaxis={
            "title": "Normalized Y coordinate",
            "range": [0, 1],
            "scaleanchor": "x",
            "scaleratio": 1,
            "constrain": "domain",
        },
        width=800,
        height=800,
        hovermode="closest",
        template="plotly_white",
    )

    return fig


def _get_fixation_statistics(fixations: pl.DataFrame) -> tuple[int, float, float]:
    count = fixations.height

    if not count:
        return 0, 0.0, 0.0

    total_duration = fixations[Column.Duration].sum() or 0.0
    average_duration = total_duration / count

    return count, total_duration, average_duration


def _add_time_series(
    fig: go.Figure, df: pl.DataFrame, column: str, name: str, color: str, *, visible: bool | str = True
):
    if column not in df.columns:
        return

    fig.add_trace(
        go.Scattergl(
            x=df[Column.Timestamp].to_list(),
            y=df[column].to_list(),
            mode="lines",
            name=name,
            line={"color": color},
            connectgaps=False,
            visible=visible,
        )
    )


def _get_color(method: str = "default") -> str:
    if method not in _COLORS:
        logging.warning("No color defined for '%s'; using default color.", method)
    return _COLORS.get(method, _COLORS["default"])


_COLORS = {
    "left_eye": "dodgerblue",
    "right_eye": "darkorange",
    "fused": "mediumseagreen",
    "fused_x": "mediumseagreen",
    "fused_y": "mediumpurple",
    "any": "mediumseagreen",
    "both": "lightseagreen",
    "aligned": "mediumpurple",
    "left": "tomato",
    "right": "orange",
    "gaze_trajectory": "silver",
    "fixations": "tomato",
    "default": "darkslategray",
}
