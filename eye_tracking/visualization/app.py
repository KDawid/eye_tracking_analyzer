import base64
from pathlib import Path

import polars as pl
from dash import Dash, Input, Output, dcc, html

from eye_tracking.columns import Column
from eye_tracking.fixation.eyefeatures import idt, ihmm, ivt
from eye_tracking.fuse_eyes import EyeFusionProtocol, fuse_raw_data
from eye_tracking.visualization.figures import (
    fixation_figure,
    gaze_trajectory_figure,
    gaze_x_figure,
    gaze_y_figure,
    pupil_diameter_figure,
)


def create_app(raw: pl.DataFrame, background_image: Path | None = None) -> Dash:

    app = Dash(__name__)

    raw_input = _is_raw_binocular_data(raw)
    fused_input = _is_fused_gaze(raw) and not raw_input
    background_image_data = _get_background_image(background_image)

    fusion_cache = {}

    def get_cached_gaze(method: str, *, drop_missing: bool = True) -> pl.DataFrame:
        if raw_input:
            protocol = EyeFusionProtocol(method)
            cache_key = (protocol, drop_missing)
        else:
            cache_key = ("pre_fused", drop_missing)

        if cache_key not in fusion_cache:
            fusion_cache[cache_key] = _get_gaze_data(raw, method, drop_missing=drop_missing)

        return fusion_cache[cache_key]

    def get_all_cached_gaze() -> dict[str, pl.DataFrame]:
        if fused_input:
            return {EyeFusionProtocol.Any.value: get_cached_gaze(EyeFusionProtocol.Any.value)}

        excluded_protocols = {EyeFusionProtocol.Left, EyeFusionProtocol.Right}
        return {
            protocol.value: get_cached_gaze(protocol.value, drop_missing=False)
            for protocol in EyeFusionProtocol
            if protocol not in excluded_protocols
        }

    app.layout = html.Div(
        [
            html.H1("Eye Tracking Analysis"),
            html.Div(
                [
                    html.H3("Fusion method"),
                    dcc.Dropdown(
                        id="fusion-method",
                        options=[{"label": protocol.label, "value": protocol.value} for protocol in EyeFusionProtocol],
                        value=EyeFusionProtocol.Any.value,
                        clearable=False,
                        disabled=fused_input,
                    ),
                    html.Div(
                        "Input data is already fused. Fusion selection is disabled.",
                        style={"display": "block" if fused_input else "none"},
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Gaze data",
                        children=[
                            dcc.Loading(
                                html.Div(
                                    [
                                        dcc.Graph(id="gaze-x-graph"),
                                        dcc.Graph(id="gaze-y-graph"),
                                        dcc.Graph(id="diameter-graph"),
                                        _background_image_checkbox(background_image_data, "show-background-image-gaze"),
                                        dcc.Graph(id="trajectory-graph"),
                                    ]
                                ),
                                type="circle",
                                fullscreen=False,
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="I-VT",
                        children=[
                            html.Div(
                                [
                                    html.H3("I-VT parameters"),
                                    html.Label("Velocity threshold"),
                                    dcc.Slider(id="ivt-threshold", min=0.0001, max=0.05, step=0.0001, value=0.001),
                                    html.Label("Minimum duration (ms)"),
                                    dcc.Slider(id="ivt-duration", min=10, max=500, step=10, value=100),
                                    html.Label("Distance metric"),
                                    dcc.Dropdown(
                                        id="ivt-distance",
                                        options=[{"label": x, "value": x} for x in ("euc", "manhattan", "chebyshev")],
                                        value="euc",
                                        clearable=False,
                                    ),
                                    _background_image_checkbox(background_image_data, "show-background-image-ivt"),
                                    dcc.Graph(id="ivt-graph"),
                                ]
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="I-DT",
                        children=[
                            html.Div(
                                [
                                    html.H3("I-DT parameters"),
                                    html.Label("Minimum duration (ms)"),
                                    dcc.Slider(id="idt-min-duration", min=10, max=500, step=10, value=100),
                                    html.Label("Maximum duration (ms)"),
                                    dcc.Slider(id="idt-max-duration", min=100, max=2000, step=50, value=1000),
                                    html.Label("Maximum dispersion"),
                                    dcc.Slider(id="idt-dispersion", min=0.001, max=0.2, step=0.001, value=0.05),
                                    html.Label("Distance metric"),
                                    dcc.Dropdown(
                                        id="idt-distance",
                                        options=[{"label": x, "value": x} for x in ("euc", "manhattan", "chebyshev")],
                                        value="euc",
                                        clearable=False,
                                    ),
                                    _background_image_checkbox(background_image_data, "show-background-image-idt"),
                                    dcc.Graph(id="idt-graph"),
                                ]
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="I-HMM",
                        children=[
                            html.Div(
                                [
                                    html.H3("I-HMM parameters"),
                                    html.Label("Fixation to saccade probability"),
                                    dcc.Slider(id="ihmm-fix2sac", min=0.01, max=0.5, step=0.01, value=0.05),
                                    html.Label("Saccade to fixation probability"),
                                    dcc.Slider(id="ihmm-sac2fix", min=0.01, max=0.5, step=0.01, value=0.05),
                                    html.Label("Distribution parameters"),
                                    dcc.Dropdown(
                                        id="ihmm-params",
                                        options=[
                                            {"label": "Default", "value": "auto"},
                                            {"label": "Estimate from data", "value": "fit"},
                                        ],
                                        value="auto",
                                        clearable=False,
                                    ),
                                    _background_image_checkbox(background_image_data, "show-background-image-ihmm"),
                                    dcc.Graph(id="ihmm-graph"),
                                ]
                            )
                        ],
                    ),
                ]
            ),
        ],
        style={"padding": "20px"},
    )

    # --------------------------------------------------
    # Fusion callback
    # --------------------------------------------------

    @app.callback(
        Output("gaze-x-graph", "figure"),
        Output("gaze-y-graph", "figure"),
        Output("diameter-graph", "figure"),
        Output("trajectory-graph", "figure"),
        Input("fusion-method", "value"),
        Input("show-background-image-gaze", "value"),
    )
    def update_gaze_graphs(method, show_background):
        all_fused = get_all_cached_gaze()
        selected_fused = get_cached_gaze(method, drop_missing=False)

        image = background_image_data if "show" in (show_background or []) else None

        return (
            gaze_x_figure(raw, all_fused, selected_method=method),
            gaze_y_figure(raw, all_fused, selected_method=method),
            pupil_diameter_figure(raw, all_fused, selected_method=method),
            gaze_trajectory_figure(selected_fused, method, background_image=image),
        )

    # --------------------------------------------------
    # I-VT callback
    # --------------------------------------------------

    @app.callback(
        Output("ivt-graph", "figure"),
        Input("fusion-method", "value"),
        Input("ivt-threshold", "value"),
        Input("ivt-duration", "value"),
        Input("ivt-distance", "value"),
        Input("show-background-image-ivt", "value"),
    )
    def update_ivt(method, threshold, duration, distance, show_background):
        gaze = get_cached_gaze(method)

        image = background_image_data if "show" in (show_background or []) else None

        fixations = ivt(gaze, velocity_threshold=threshold, min_duration=duration, distance=distance)

        return fixation_figure(
            gaze,
            fixations,
            algorithm="I-VT",
            parameters={"threshold": threshold, "duration": duration, "distance": distance},
            background_image=image,
        )

    # --------------------------------------------------
    # I-DT callback
    # --------------------------------------------------

    @app.callback(
        Output("idt-graph", "figure"),
        Input("fusion-method", "value"),
        Input("idt-min-duration", "value"),
        Input("idt-max-duration", "value"),
        Input("idt-dispersion", "value"),
        Input("idt-distance", "value"),
        Input("show-background-image-idt", "value"),
    )
    def update_idt(method, min_duration, max_duration, dispersion, distance, show_background):
        gaze = get_cached_gaze(method)

        image = background_image_data if "show" in (show_background or []) else None

        if max_duration <= min_duration:
            return fixation_figure(
                gaze, pl.DataFrame(), algorithm="I-DT (invalid parameters)", parameters={}, background_image=image
            )

        fixations = idt(
            gaze, min_duration=min_duration, max_duration=max_duration, max_dispersion=dispersion, distance=distance
        )

        return fixation_figure(
            gaze,
            fixations,
            algorithm="I-DT",
            parameters={
                "min_duration": min_duration,
                "max_duration": max_duration,
                "dispersion": dispersion,
                "distance": distance,
            },
            background_image=image,
        )

    # --------------------------------------------------
    # I-HMM callback
    # --------------------------------------------------

    @app.callback(
        Output("ihmm-graph", "figure"),
        Input("fusion-method", "value"),
        Input("ihmm-fix2sac", "value"),
        Input("ihmm-sac2fix", "value"),
        Input("ihmm-params", "value"),
        Input("show-background-image-ihmm", "value"),
    )
    def update_ihmm(method, fix2sac, sac2fix, params, show_background):
        gaze = get_cached_gaze(method)

        image = background_image_data if "show" in (show_background or []) else None

        fixations = ihmm(
            gaze,
            fixation_to_saccade_probability=fix2sac,
            saccade_to_fixation_probability=sac2fix,
            distribution_parameters=params,
        )

        return fixation_figure(
            gaze,
            fixations,
            algorithm="I-HMM",
            parameters={
                "fixation_to_saccade_probability": fix2sac,
                "saccade_to_fixation_probability": sac2fix,
                "distribution_parameters": params,
            },
            background_image=image,
        )

    return app


def _background_image_checkbox(background_image_data: str | None, component_id: str):
    return dcc.Checklist(
        id=component_id,
        options=[{"label": " Show background image", "value": "show"}],
        value=[],
        style={"display": "block" if background_image_data else "none", "margin": "10px 0"},
    )


def _is_fused_gaze(df: pl.DataFrame) -> bool:
    return all(column in df.columns for column in (Column.Timestamp, Column.EyeX, Column.EyeY))


def _is_raw_binocular_data(df: pl.DataFrame) -> bool:
    return all(
        column in df.columns for column in (Column.Timestamp, Column.LeftX, Column.LeftY, Column.RightX, Column.RightY)
    )


def _get_gaze_data(raw: pl.DataFrame, method: str | EyeFusionProtocol, *, drop_missing: bool = True) -> pl.DataFrame:
    if _is_raw_binocular_data(raw):
        protocol = EyeFusionProtocol(method)
        return fuse_raw_data(raw, protocol, drop_missing=drop_missing)

    if _is_fused_gaze(raw):
        columns = [Column.Timestamp, Column.EyeX, Column.EyeY]
        if Column.Diameter in raw.columns:
            columns.append(Column.Diameter)
        return raw.select(columns)

    raise ValueError("Input must contain binocular raw data or already-fused gaze data.")


def _get_background_image(image_path: Path | None) -> str | None:
    if image_path is None:
        return None

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Background image not found: {image_path}")

    if image_path.suffix.lower() != ".png":
        raise ValueError("Background image must be a PNG file.")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


if __name__ == "__main__":
    from eye_tracking.input_formats import MATLAB, SIMPLEGAZETRACKER, TOBII
    from eye_tracking.reader import read_demo_data, read_matlab_data, read_raw_data

    # raw = read_demo_data("../../notebooks/data/demo_data.txt")
    # raw = read_raw_data(Path("../../notebooks/simplegazetracker_dummy.csv"), fmt=SIMPLEGAZETRACKER)
    # raw = read_raw_data(Path("../../notebooks/data/1_rawdata.csv"), fmt=TOBII).slice(3200, 300)
    # raw = read_matlab_data(Path("../../local_notes/UH47_img_Europe_labelled_MN.mat"), fmt=MATLAB)
    raw = read_matlab_data(Path("../../local_notes/UH29_img_Europe_labelled_FIX_MN.mat"), fmt=MATLAB)
    # raw = read_matlab_data(Path("../../local_notes/UH29_img_Europe_labelled_RA.mat"), fmt=MATLAB)

    background_image = Path("../../local_notes/Europe1024x768.png")

    app = create_app(raw, background_image)
    app.run(debug=True)
