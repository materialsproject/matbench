"""Plotting utils for CrabNet Hyperparameter optimization."""
from typing import Iterable, List
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from ax.modelbridge import ModelBridge
from ax.plot.helper import compose_annotation
from ax.utils.common.logger import get_logger

logger = get_logger(__name__)


def matplotlibify(fig, size=24, width_inches=3.5, height_inches=3.5, dpi=142):
    # make it look more like matplotlib
    # modified from: https://medium.com/swlh/formatting-a-plotly-figure-with-matplotlib-style-fa56ddd97539)
    font_dict = dict(family="Arial", size=size, color="black")

    fig.update_layout(
        font=font_dict,
        plot_bgcolor="white",
        width=width_inches * dpi,
        height=height_inches * dpi,
        margin=dict(r=40, t=20, b=10),
    )

    fig.update_yaxes(
        showline=True,  # add line at x=0
        linecolor="black",  # line color
        linewidth=2.4,  # line size
        ticks="inside",  # ticks outside axis
        tickfont=font_dict,  # tick label font
        mirror="allticks",  # add ticks to top/right axes
        tickwidth=2.4,  # tick width
        tickcolor="black",  # tick color
    )

    fig.update_xaxes(
        showline=True,
        showticklabels=True,
        linecolor="black",
        linewidth=2.4,
        ticks="inside",
        tickfont=font_dict,
        mirror="allticks",
        tickwidth=2.4,
        tickcolor="black",
    )
    fig.update(layout_coloraxis_showscale=False)

    width_default_px = fig.layout.width
    targ_dpi = 300
    scale = width_inches / (width_default_px / dpi) * (targ_dpi / dpi)

    return fig, scale


def my_plot_feature_importance_by_feature_plotly(
    model: ModelBridge = None,
    feature_importances: dict = None,
    error_x: dict = None,
    metric_names: Iterable[str] = None,
    relative: bool = True,
    caption: str = "",
) -> go.Figure:
    """One plot per metric, showing importances by feature.

    Args:
        model: A model with a ``feature_importances`` method.
        relative: whether to normalize feature importances so that they add to 1.
        caption: an HTML-formatted string to place at the bottom of the plot.

    Returns a go.Figure of feature importances.
    
    Notes:
        Copyright (c) Facebook, Inc. and its affiliates.

        This source code is licensed under the MIT license. Modifed by @sgbaird.
    """
    traces = []
    dropdown = []
    if metric_names is None:
        assert model is not None, "specify model or metric_names"
        metric_names = model.metric_names
    assert metric_names is not None, "specify model or metric_names"
    for i, metric_name in enumerate(sorted(metric_names)):
        try:
            if feature_importances is not None:
                importances = feature_importances
            else:
                assert model is not None, "model is None"
                importances = model.feature_importances(metric_name)
        except NotImplementedError:
            logger.warning(
                f"Model for {metric_name} does not support feature importances."
            )
            continue
        factor_col = "Factor"
        importance_col = "Importance"
        std_col = "StdDev"
        low_col = "err_minus"
        assert error_x is not None, "specify error_x"
        df = pd.DataFrame(
            [
                {factor_col: factor, importance_col: importance}
                for factor, importance in importances.items()
            ]
        )
        err_df = pd.Series(error_x).to_frame(name=std_col)
        err_df.index.names = [factor_col]
        df = pd.concat((df.set_index(factor_col), err_df), axis=1).reset_index()

        if relative:
            totals = df[importance_col].sum()
            df[importance_col] = df[importance_col].div(totals)
            df[std_col] = df[std_col].div(totals)

        low_df = df[std_col]
        low_df[low_df > df[importance_col]] = df[importance_col]
        df[low_col] = low_df

        df = df.sort_values(importance_col)
        traces.append(
            go.Bar(
                name=importance_col,
                orientation="h",
                visible=i == 0,
                x=df[importance_col],
                y=df[factor_col],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=df[std_col].to_list(),
                    arrayminus=df[low_col].to_list(),
                ),
            )
        )

        is_visible = [False] * len(sorted(metric_names))
        is_visible[i] = True
        dropdown.append(
            {"args": ["visible", is_visible], "label": metric_name, "method": "restyle"}
        )
    if not traces:
        raise NotImplementedError("No traces found for metric")

    updatemenus = [
        {
            "x": 0,
            "y": 1,
            "yanchor": "top",
            "xanchor": "left",
            "buttons": dropdown,
            "pad": {
                "t": -40
            },  # hack to put dropdown below title regardless of number of features
        }
    ]
    features = traces[0].y
    title = (
        "Relative Feature Importances" if relative else "Absolute Feature Importances"
    )
    layout = go.Layout(
        height=200 + len(features) * 20,
        hovermode="closest",
        margin=go.layout.Margin(
            l=8 * min(max(len(idx) for idx in features), 75)  # noqa E741
        ),
        showlegend=False,
        title=title,
        updatemenus=updatemenus,
        annotations=compose_annotation(caption=caption),
    )

    if relative:
        layout.update({"xaxis": {"tickformat": ".0%"}})

    fig = go.Figure(data=traces, layout=layout)

    return fig
