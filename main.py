from torch.optim import Optimizer, Adam
import torch

from Supervisor import Supervisor
from Embedding import Embedding
from Generator import Generator
from Discriminator import Discriminator
from Recovery import Recovery
from TimeGAN import _recovery_forward, _supervisor_forward, _generator_forward, _discriminator_forward, _inference
from utils import plot_time_series
import numpy as np
from typing import Dict
from torch.utils.data import DataLoader
from data import GeneralDataset
import yaml
import wandb

'''
    
    Trainers
    
'''


def embedding_trainer(emb: Embedding,
                      sup: Supervisor,
                      rec: Recovery,
                      emb_opt: Optimizer,
                      rec_opt: Optimizer,
                      dl: DataLoader,
                      cfg: Dict) -> None:
    num_epochs = int(cfg['emb']['num_epochs'])
    device = torch.device(cfg['system']['device'])

    for epoch in range(num_epochs):
        for idx, real_data in enumerate(dl):
            x, t = real_data

            x = x.float()
            x = x.view(*x.shape, 1)
            x = x.to(device)

            t = t.view(-1)

            # Reset gradients
            emb.zero_grad()
            rec.zero_grad()
            sup.zero_grad()

            # Forward Pass
            _, e_loss0, e_loss_t0 = _recovery_forward(emb=emb, sup=sup, rec=rec, x=x, t=t)
            loss = np.sqrt(e_loss_t0.item())

            # Backward Pass
            e_loss0.backward()

            # Update model parameters
            emb_opt.step()
            rec_opt.step()

        print(f"[EMB] Epoch: {epoch}, Loss: {loss:.4f}")


def supervisor_trainer(emb: Embedding,
                       sup: Supervisor,
                       sup_opt: Optimizer,
                       dl: DataLoader,
                       cfg: Dict) -> None:
    num_epochs = int(cfg['sup']['num_epochs'])
    device = torch.device(cfg['system']['device'])

    for epoch in range(num_epochs):
        for idx, real_data in enumerate(dl):
            x, t = real_data

            x = x.float()
            x = x.view(*x.shape, 1)
            x = x.to(device)
            t = t.view(-1)

            # Reset gradients
            emb.zero_grad()
            sup.zero_grad()

            # Forward Pass
            sup_loss = _supervisor_forward(emb=emb, sup=sup, x=x, t=t)

            # Backward Pass
            sup_loss.backward()
            loss = np.sqrt(sup_loss.item())

            # Update model parameters
            sup_opt.step()

        print(f"[SUP] Epoch: {epoch}, Loss: {loss:.4f}")


def joint_trainer(emb: Embedding,
                  sup: Supervisor,
                  g: Generator,
                  d: Discriminator,
                  rec: Recovery,
                  g_opt: Optimizer,
                  d_opt: Optimizer,
                  sup_opt: Optimizer,
                  rec_opt: Optimizer,
                  emb_opt: Optimizer,
                  dl: DataLoader,
                  cfg: Dict) -> None:
    num_epochs = int(cfg['system']['jointly_num_epochs'])
    batch_size = int(cfg['system']['batch_size'])
    seq_len = int(cfg['system']['seq_len'])
    dim_latent = int(cfg['g']['dim_latent'])
    d_threshold = float(cfg['d']['threshold'])
    device = torch.device(cfg['system']['device'])

    for epoch in range(num_epochs):
        for idx, real_data in enumerate(dl):
            x, t = real_data

            x = x.float()
            x = x.view(*x.shape, 1)
            x = x.to(device)
            t = t.view(-1)

            # Generator Training
            for _ in range(2):
                # Random sequence
                z = torch.rand((batch_size, seq_len, dim_latent))

                # Forward Pass (Generator)
                emb.zero_grad()
                rec.zero_grad()
                sup.zero_grad()
                g.zero_grad()
                d.zero_grad()

                g_loss = _generator_forward(emb=emb, sup=sup, rec=rec, g=g, d=d, x=x, t=t, z=z)
                g_loss.backward()
                g_loss = np.sqrt(g_loss.item())

                # Update model parameters
                g_opt.step()
                sup_opt.step()

                # Forward Pass (Embedding)
                emb.zero_grad()
                rec.zero_grad()
                sup.zero_grad()

                e_loss, _, e_loss_t0 = _recovery_forward(emb=emb, rec=rec, sup=sup, x=x, t=t)
                e_loss.backward()
                e_loss = np.sqrt(e_loss.item())

                # Update model parameters
                emb_opt.step()
                rec_opt.step()

            # Random sequence
            z = torch.rand((batch_size, seq_len, dim_latent)).to(device)

            # Discriminator Training
            emb.zero_grad()
            sup.zero_grad()
            g.zero_grad()
            d.zero_grad()

            # Forward Pass
            d_loss = _discriminator_forward(emb=emb, sup=sup, g=g, d=d, x=x, t=t, z=z)

            # Check Discriminator loss
            if d_loss > d_threshold:
                # Backward Pass
                d_loss.backward()

                # Update model parameters
                d_opt.step()
            d_loss = d_loss.item()

            if idx == len(dl) - 1:
                # Generate sample
                sample = _inference(sup=sup, g=g, rec=rec, z=z, t=t)
                fake_sample = plot_time_series( sample.detach().cpu().numpy()[0].reahpe(seq_len),
                                                'Generated sample {}'.format(epoch))
                real_sample = fig = plot_time_series(x.detach().cpu().numpy()[0].reshape(seq_len),
                                                     'Real sample {}'.format(epoch))
                wandb.log({
                        "epoch": epoch,
                        "d loss": d_loss.detach().cpu(),
                        "g loss": g_loss.detach().cpu().item(),
                        "e loss": e_loss.detach().cpu().item(),
                        "Fake sample": fake_sample,
                        "Real sample": real_sample
                    })

        print(f"[JOINT] Epoch: {epoch}, E_loss: {e_loss:.4f}, G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f}")


def time_gan_trainer(cfg: Dict) -> None:
    # Init all parameters and models
    dataset_name = cfg['system']['dataset']
    seq_len = int(cfg['system']['seq_len'])
    batch_size = int(cfg['system']['batch_size'])
    model_name = cfg['system']['model_name']
    device = torch.device(cfg['system']['device'])
    lr = float(cfg['system']['lr'])

    ds_generator = GeneralDataset.GeneralDataset(seq_len, dataset_name, model_name)
    ds = ds_generator.get_dataset()

    dl = DataLoader(ds, num_workers=10, batch_size=batch_size, shuffle=True)

    # TimeGAN elements
    emb = Embedding(cfg=cfg).to(device)
    rec = Recovery(cfg=cfg).to(device)
    sup = Supervisor(cfg=cfg).to(device)
    g = Generator(cfg=cfg).to(device)
    d = Discriminator(cfg=cfg).to(device)

    # Optimizers
    # TODO: see the behaviour and update lr with TTsUR if necessary

    emb_opt = Adam(emb.parameters(), lr=lr)
    rec_opt = Adam(rec.parameters(), lr=lr)
    sup_opt = Adam(sup.parameters(), lr=lr)
    g_opt = Adam(g.parameters(), lr=lr)
    d_opt = Adam(d.parameters(), lr=lr)

    print(f"[EMB] Start Embedding network training")
    embedding_trainer(emb=emb, rec=rec, sup=sup, emb_opt=emb_opt, rec_opt=rec_opt, dl=dl, cfg=cfg)

    print(f"[SUP] Start Supervisor network training")
    supervisor_trainer(emb=emb, sup=sup, sup_opt=sup_opt, dl=dl, cfg=cfg)

    print(f"[JOINT] Start joint training")
    joint_trainer(emb=emb,
                  rec=rec,
                  sup=sup,
                  g=g,
                  d=d,
                  emb_opt=emb_opt,
                  rec_opt=rec_opt,
                  sup_opt=sup_opt,
                  g_opt=g_opt,
                  d_opt=d_opt,
                  dl=dl,
                  cfg=cfg)

    # Move models to cpu
    emb = emb.to('cpu')
    rec = rec.to('cpu')
    sup = sup.to('cpu')
    g = g.to('cpu')
    d = d.to('cpu')

    torch.save(emb.state_dict(), './trained_models/emb.pt')
    torch.save(rec.state_dict(), './trained_models/rec.pt')
    torch.save(sup.state_dict(), './trained_models/sup.pt')
    torch.save(g.state_dict(), './trained_models/g.pt')
    torch.save(d.state_dict(), './trained_models/d.pt')


if __name__ == '__main__':
    torch.random.manual_seed(42)

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    run_name = config['system']['run_name'] + ' ' + config['system']['dataset']
    wandb.init(config=config, project='_timegan_baseline_', name=run_name)

    time_gan_trainer(cfg=config)
