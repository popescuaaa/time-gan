import numpy as np
import plotly.graph_objects as go


def plot_time_series(ts: np.ndarray, figure_name):
    time = [i for i in range(len(ts))]
    fig = go.Figure(data=go.Scatter(x=time, y=ts, name=figure_name))
    fig.update_layout(
        autosize=False,
        width=230,
        height=230,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        paper_bgcolor="LightSteelBlue",
    )
    return fig
