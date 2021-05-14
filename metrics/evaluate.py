from Supervisor import Supervisor
from Recovery import Recovery
from Generator import Generator
from Embedding import Embedding
from TimeGAN import _supervisor_forward, _embedding_forward_side, _inference
import numpy as np
from torch import Tensor
import torch
from typing import Dict
import wandb
from visualisation import visualize
from Energy import extract_time, Energy


def plot_all_samples(real_samples: np.ndarray,
                     sup: Supervisor,
                     rec: Recovery,
                     g: Generator,
                     time: Tensor,
                     device: torch.device,
                     data_shape: torch.Size,
                     perplexity: int):
    real_samples_tensor = torch.from_numpy(np.array(real_samples[:1000]))
    real_samples_tensor = real_samples_tensor.view(real_samples_tensor.shape[0],
                                                   real_samples_tensor.shape[1] * \
                                                   real_samples_tensor.shape[2])
    # Generate a balanced distribution
    generated_samples = []
    with torch.no_grad():
        for _ in range(len(real_samples[:1000])):
            _z = torch.rand(data_shape).to(device)
            sample = _inference(sup=sup, g=g, rec=rec, z=_z, t=time)
            generated_samples.append(sample.detach().cpu().numpy()[0, :, :])

    generated_samples_tensor = torch.from_numpy(np.array(generated_samples))
    generated_samples_tensor = generated_samples_tensor.view(generated_samples_tensor.shape[0],
                                                             generated_samples_tensor.shape[1] * \
                                                             generated_samples_tensor.shape[2])

    dist_fig = visualize(real_data=real_samples_tensor.numpy(),
                         generated_data=generated_samples_tensor.numpy(),
                         perplexity=perplexity)

    return dist_fig


def plot_sup_samples(emb: Embedding,
                     sup: Supervisor,
                     device: torch.device,
                     real_samples: np.ndarray,
                     batch_size: int,
                     perplexity: int):
    generated_samples = []
    target_samples = []

    with torch.no_grad():
        for e in real_samples[:1000]:
            e_tensor = torch.from_numpy(e).repeat(batch_size, 1, 1).float()
            e_tensor = e_tensor.to(device)
            e_tensor = e_tensor.float()
            _t, _ = extract_time(e_tensor)
            _, h, _h_sup = _supervisor_forward(emb=emb, sup=sup, x=e_tensor, t=_t)
            generated_samples.append(_h_sup.detach().cpu().numpy()[0, :, :])
            target_samples.append(h.detach().cpu().numpy()[0, :, :])

    generated_samples_tensor = torch.from_numpy(np.array(generated_samples))
    generated_samples_tensor = generated_samples_tensor.view(generated_samples_tensor.shape[0],
                                                             generated_samples_tensor.shape[1] * \
                                                             generated_samples_tensor.shape[2])

    target_samples_tensor = torch.from_numpy(np.array(target_samples))
    target_samples_tensor = generated_samples_tensor.view(target_samples_tensor.shape[0],
                                                          target_samples_tensor.shape[1] * \
                                                          target_samples_tensor.shape[2])

    fig = visualize(real_data=target_samples_tensor.numpy(),
                    generated_data=generated_samples_tensor.numpy(),
                    perplexity=perplexity)

    return fig


def plot_emb_samples(real_samples: np.ndarray,
                     batch_size: int,
                     device: torch.device,
                     emb: Embedding,
                     rec: Recovery,
                     perplexity: int):
    real_samples_tensor = torch.from_numpy(np.array(real_samples[:1000]))
    real_samples_tensor = real_samples_tensor.view(real_samples_tensor.shape[0],
                                                   real_samples_tensor.shape[1] * \
                                                   real_samples_tensor.shape[2])

    generated_samples = []
    with torch.no_grad():
        for e in real_samples[:1000]:
            e_tensor = torch.from_numpy(e).repeat(batch_size, 1, 1).float()
            e_tensor = e_tensor.to(device)
            e_tensor = e_tensor.float()
            _t, _ = extract_time(e_tensor)
            _, sample = _embedding_forward_side(emb=emb, rec=rec, x=e_tensor, t=_t)
            generated_samples.append(sample.detach().cpu().numpy()[0, :, :])

    generated_samples_tensor = torch.from_numpy(np.array(generated_samples))
    generated_samples_tensor = generated_samples_tensor.view(generated_samples_tensor.shape[0],
                                                             generated_samples_tensor.shape[1] * \
                                                             generated_samples_tensor.shape[2])

    fig = visualize(real_data=real_samples_tensor.numpy(),
                                  generated_data=generated_samples_tensor.numpy(),
                                  perplexity=perplexity)

    return fig


def evaluate(cfg: Dict, LOGGING_STEP: int) -> None:
    seq_len = int(cfg['system']['seq_len'])
    batch_size = int(cfg['system']['batch_size'])
    device = torch.device(cfg['system']['device'])

    ds = Energy(seq_len)

    # Load TimeGAN elements
    emb = Embedding(cfg=cfg)
    emb.load_state_dict(torch.load('./trained_models/emb.pt'))
    emb.eval()

    rec = Recovery(cfg=cfg)
    rec.load_state_dict(torch.load('./trained_models/rec.pt'))
    rec.eval()

    sup = Supervisor(cfg=cfg)
    sup.load_state_dict(torch.load('./trained_models/sup.pt'))
    sup.eval()

    g = Generator(cfg=cfg)
    g.load_state_dict(torch.load('./trained_models/g.pt'))
    g.eval()

    emb = emb.to(device)
    rec = rec.to(device)
    sup = sup.to(device)
    g = g.to(device)

    real_samples = ds.get_distribution()
    time = torch.from_numpy(np.array([seq_len] * batch_size))
    data_shape = torch.Size((batch_size, seq_len, emb.dim_features))

    print('[EVAL] Plotting emb samples')
    emb_samples = plot_emb_samples(real_samples=real_samples,
                                   emb=emb,
                                   rec=rec,
                                   device=device,
                                   batch_size=batch_size,
                                   perplexity=40)

    print('[EVAL] Plotting sup samples')
    sup_samples = plot_sup_samples(emb,
                                   sup,
                                   device,
                                   real_samples,
                                   batch_size,
                                   perplexity=40)

    print('[EVAL] Plotting all samples')
    all_samples = plot_all_samples(real_samples=real_samples,
                                   g=g, sup=sup,
                                   rec=rec,
                                   time=time,
                                   data_shape=data_shape,
                                   perplexity=40,
                                   device=device)

    LOGGING_STEP += 1
    wandb.log({
        'All samples': all_samples,
        'Emb samples': emb_samples,
        'Sup samples': sup_samples
    }, step=LOGGING_STEP)
