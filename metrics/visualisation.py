import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np


def visualize(generated_data: np.ndarray,
              real_data: np.ndarray,
              perplexity: int,
              legend: [str]):
    no, seq_len, dim = real_data.shape

    for i in range(no):
        if i == 0:
            processed_data = np.reshape(np.mean(real_data[0, :, :], 1), [1, seq_len])
            processed_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            processed_data = np.concatenate((processed_data,
                                             np.reshape(np.mean(real_data[i, :, :], 1), [1, seq_len])))

            processed_data_hat = np.concatenate((processed_data_hat,
                                                 np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    processed_data_t_sne = np.concatenate((processed_data, processed_data_hat), axis=0)

    # t-SNE
    t_sne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=300)
    t_sne_results = t_sne.fit_transform(processed_data_t_sne)

    # PCA
    pca = PCA(n_components=2)
    pca.fit(processed_data)
    pca_results = pca.transform(processed_data)
    pca_hat_results = pca.transform(processed_data_hat)

    size = len(real_data)
    generated_x = t_sne_results[size:, 0]
    generated_y = t_sne_results[size:, 1]

    real_x = t_sne_results[:size, 0]
    real_y = t_sne_results[:size, 1]

    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(go.Scatter(x=generated_x, y=generated_y,
                             mode='markers',
                             opacity=0.3,
                             name=legend[0] + ' t-SNE',
                             marker={
                                 'size': 10,
                                 'color': 'red'
                             }),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=real_x, y=real_y,
                             mode='markers',
                             opacity=0.3,
                             name=legend[1] + ' t-SNE',
                             marker={
                                 'size': 10,
                                 'color': 'blue'
                             }),
                  row=1, col=1)

    generated_x = pca_hat_results[:, 0]
    generated_y = pca_hat_results[:, 1]

    real_x = pca_results[:, 0]
    real_y = pca_results[:, 1]

    fig.add_trace(go.Scatter(x=generated_x, y=generated_y,
                             mode='markers',
                             opacity=0.3,
                             name=legend[0] + ' PCA',
                             marker={
                                 'size': 10,
                                 'color': 'red'
                             }),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=real_x, y=real_y,
                             mode='markers',
                             opacity=0.3,
                             name=legend[1] + ' PCA',
                             marker={
                                 'size': 10,
                                 'color': 'blue'
                             }),
                  row=1, col=2)

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
        paper_bgcolor='LightSteelBlue',
        title_text='t-SNE and PCA data representations')

    return fig
