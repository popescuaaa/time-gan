import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


"""
    Plot reconstructed data from embedding training
"""


def plot_two_time_series(real: np.ndarray, reconstructed: np.ndarray):
    time = [i for i in range(len(real))]
    trace1 = go.Scatter(x=time, y=real, name="Real time series")
    trace2 = go.Scatter(x=time, y=reconstructed, name="Reconstructed time series")
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)
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
