import torch
from torch import Tensor
from typing import Tuple, Any
import torch.nn.functional as F
from TEmbedding import TEmbedding
from TRecovery import TRecoveryEncoder
from TDiscriminator import TDiscriminatorEncoder
from TSupervisor import TSupervisor
from TGenerator import TGenerator


def _embedding_forward_side(emb: TEmbedding,
                            rec: TRecoveryEncoder,
                            src: Tensor) -> Tuple[Tensor, Tensor]:
    assert src.device == emb.device, 'Src and EMB are not on the same device'
    h = emb(src)
    _src = rec(src)
    e_loss_t0 = F.mse_loss(_src, src)
    e_loss0 = 10 * torch.sqrt(e_loss_t0)
    return e_loss0, _src


def _embedding_forward_main(emb: TEmbedding,
                            rec: TRecoveryEncoder,
                            sup: TSupervisor,
                            src: Tensor) -> Tuple[Any, Any, Tensor]:
    assert src.device == emb.device, 'Src and EMB are not on the same device'
    h = emb(src)
    _src = rec(src)
    _h_sup = sup(src, h)

    g_loss_sup = F.mse_loss(
        _h_sup[:, :-1, :],
        h[:, 1:, :]
    )

    # Reconstruction Loss
    e_loss_t0 = F.mse_loss(_src, src)
    e_loss0 = 10 * torch.sqrt(e_loss_t0)
    e_loss = e_loss0 + 0.1 * g_loss_sup
    return e_loss, e_loss0, e_loss_t0


def _supervisor_forward(emb: TEmbedding,
                        sup: TSupervisor,
                        src: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    assert src.device == emb.device, 'Src and EMB are not on the same device'

    # Supervisor forward pass
    h = emb(src)
    _h_sup = sup(src, h)

    # Supervised loss

    # Teacher forcing next output
    s_loss = F.mse_loss(
        _h_sup[:, :-1, :],
        h[:, 1:, :]
    )

    return s_loss, h, _h_sup


def _discriminator_forward(emb: TEmbedding,
                           sup: TSupervisor,
                           g: TGenerator,
                           d: TDiscriminatorEncoder,
                           src: Tensor,
                           z: Tensor,
                           gamma=1.0) -> Tensor:
    assert src.device == emb.device, 'Src and EMB are not on the same device'
    assert z.device == g.device, 'z and G are not on the same device'

    # Discriminator forward pass and adversarial loss
    h = emb(src).detach()
    _h = sup(src, h).detach()
    _e = g(z).detach()

    # Forward Pass
    y_real = d(h)  # Encoded original data
    y_fake = d(_h)  # Output of supervisor
    y_fake_e = d(_e)  # Output of generator

    d_loss_real = F.binary_cross_entropy_with_logits(y_real, torch.ones_like(y_real))
    d_loss_fake = F.binary_cross_entropy_with_logits(y_fake, torch.zeros_like(y_fake))
    d_loss_fake_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.zeros_like(y_fake_e))

    d_loss = d_loss_fake + d_loss_real + gamma * d_loss_fake_e

    return d_loss


def _generator_forward(emb: TEmbedding,
                       sup: TSupervisor,
                       g: TGenerator,
                       d: TDiscriminatorEncoder,
                       rec: TRecoveryEncoder,
                       src: Tensor,
                       z: Tensor,
                       gamma=1.0) -> Tensor:
    assert src.device == emb.device, 'Src and EMB are not on the same device'
    assert z.device == g.device, 'z and G are not on the same device'

    # Supervised Forward Pass
    h = emb(src)
    _h_sup = sup(src, h)
    _x = rec(h)

    # Generator Forward Pass
    _e = g(z)
    _h = sup(src, _e)

    # Synthetic generated data
    _src = rec(_h)  # recovered data

    # Generator Loss

    # 1. Adversarial loss
    y_fake = d(_h)  # Output of supervisor
    y_fake_e = d(_e)  # Output of generator

    g_loss_u = F.binary_cross_entropy_with_logits(y_fake, torch.ones_like(y_fake))
    g_loss_u_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.ones_like(y_fake_e))

    # 2. Supervised loss
    g_loss_s = torch.nn.functional.mse_loss(_h_sup[:, :-1, :], h[:, 1:, :])  # Teacher forcing next output

    # 3. Two Moments
    g_loss_v1 = torch.mean(torch.abs(
        torch.sqrt(_x.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(src.var(dim=0, unbiased=False) + 1e-6)))
    g_loss_v2 = torch.mean(torch.abs((_x.mean(dim=0)) - (src.mean(dim=0))))

    g_loss_v = g_loss_v1 + g_loss_v2

    # 4. Sum
    g_loss = g_loss_u + gamma * g_loss_u_e + 100 * torch.sqrt(g_loss_s) + 100 * g_loss_v

    return g_loss


def _inference(sup: TSupervisor,
               g: TGenerator,
               rec: TRecoveryEncoder,
               src: Tensor,
               batch_size: int,
               seq_len: int,
               z: Tensor) -> Tensor:
    # Generate synthetic data
    assert z.device == g.device, 'z and Time GAN are not on the same device'

    # Generator Forward Pass
    _e = g(z)

    """
    #let's assume batch_size = 1
    initial_dec_input = zeros(1, 1, emb_dim) #All 0s
    tgt_emb = zeros(1, tgt_size, emb_dim)
     tgt_emb[0,0, :] = initial_dec_input
    for i in range(tgt_size):
      out = model(inp_emb, tgt_emb)
      tgt_emb[0, i+1, :] = out[0,i,:]
      
      @link: https://discuss.pytorch.org/t/how-to-use-train-transformer-in-pytorch/72607/6
      @link: https://discuss.pytorch.org/t/how-to-use-nn-transformerdecoder-at-inference-time/49484/5
  
    """
    # initial_sup_input = torch.zeros(size=(batch_size, 1, g.dim_output))  # 0s
    tgt = torch.zeros_like(_e)
    # tgt[0, 0, :] = initial_sup_input
    for i in range(seq_len - 1):
        _h = sup(tgt, _e)
        tgt[0, i + 1, :] = _h[0, i, :g.dim_output]

    # Synthetic generated data (reconstructed)
    _x = rec(_h)
    return _x
