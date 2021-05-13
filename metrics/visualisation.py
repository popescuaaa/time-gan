import plotly.graph_objects as go
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pylab as plt


def visualize(generated_data: np.ndarray, real_data: np.ndarray, perplexity: int):
    # Do t-SNE Analysis together
    anal_sample_no = len(real_data)
    processed_data = np.concatenate((real_data, generated_data), axis=0)
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]
    t_sne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=300)
    t_sne_results = t_sne.fit_transform(processed_data)

    size = len(real_data)
    generated_x = t_sne_results[size:, 0]
    generated_y = t_sne_results[size:, 1]

    real_x = t_sne_results[:size, 0]
    real_y = t_sne_results[:size, 1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=generated_x, y=generated_y,
                             mode='markers',
                             name='Synthetic distribution',
                             marker_color='red'))

    fig.add_trace(go.Scatter(x=real_x, y=real_y,
                             mode='markers',
                             name='Real distribution',
                             marker_color='blue'))

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
        paper_bgcolor="LightSteelBlue")

    # f, ax = plt.subplots(1)
    #
    # plt.scatter(t_sne_results[:anal_sample_no, 0], t_sne_results[:anal_sample_no, 1],
    #             c=colors[:anal_sample_no], alpha=0.2, label="Original")
    # plt.scatter(t_sne_results[anal_sample_no:, 0], t_sne_results[anal_sample_no:, 1],
    #             c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")
    #
    # ax.legend()
    #
    # plt.title('t-SNE plot')
    # plt.xlabel('x-tsne')
    # plt.ylabel('y_tsne')

    return fig
