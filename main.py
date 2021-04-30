from torch.optim import Optimizer, Adam
import torch
from TimeGAN import TimeGAN
import numpy as np
from typing import Dict
from torch.utils.data import DataLoader
from data import GeneralDataset
import yaml
import wandb

'''
    
    Trainers
    
'''


def embedding_trainer(time_gan: TimeGAN,
                      emb_opt: Optimizer,
                      rec_opt: Optimizer,
                      dl: DataLoader,
                      cfg: Dict) -> None:
    num_epochs = int(cfg['emb']['num_epochs'])

    for epoch in range(num_epochs):
        for x, t in enumerate(dl):
            # Reset gradients
            time_gan.zero_grad()

            # Forward Pass
            _, e_loss0, e_loss_t0 = time_gan(x, t, None, "embedding")
            loss = np.sqrt(e_loss_t0.item())

            # Backward Pass
            e_loss0.backward()

            # Update model parameters
            emb_opt.step()
            rec_opt.step()

        print(f"[EMB] Epoch: {epoch}, Loss: {loss:.4f}")


def supervisor_trainer(time_gan: TimeGAN,
                       sup_opt: Optimizer,
                       dl: DataLoader,
                       cfg: Dict) -> None:
    num_epochs = int(cfg['sup']['num_epochs'])

    for epoch in range(num_epochs):
        for x, t in enumerate(dl):
            # Reset gradients
            time_gan.zero_grad()

            # Forward Pass
            sup_loss = time_gan(x, t, None, "supervisor")

            # Backward Pass
            sup_loss.backward()
            loss = np.sqrt(sup_loss.item())

            # Update model parameters
            sup_opt.step()

        print(f"[SUP] Epoch: {epoch}, Loss: {loss:.4f}")


def joint_trainer(time_gan: TimeGAN,
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

    for epoch in range(num_epochs):
        for x, t in enumerate(dl):
            # Generator Training
            for _ in range(2):
                # Random sequence
                z = torch.rand((batch_size, seq_len, dim_latent))

                # Forward Pass (Generator)
                time_gan.zero_grad()
                g_loss = time_gan(x, t, z, "generator")
                g_loss.backward()
                g_loss = np.sqrt(g_loss.item())

                # Update model parameters
                g_opt.step()
                sup_opt.step()

                # Forward Pass (Embedding)
                time_gan.zero_grad()
                e_loss, _, e_loss_t0 = time_gan(x, t, z, "embedding")
                e_loss.backward()
                e_loss = np.sqrt(e_loss.item())

                # Update model parameters
                emb_opt.step()
                rec_opt.step()

            # Random sequence
            z = torch.rand((batch_size, seq_len, dim_latent))

            # Discriminator Training
            time_gan.zero_grad()
            # Forward Pass
            d_loss = time_gan(x, t, z, "discriminator")

            # Check Discriminator loss
            if d_loss > d_threshold:
                # Backward Pass
                d_loss.backward()

                # Update model parameters
                d_opt.step()
            d_loss = d_loss.item()

        print(f"[JOINT] Epoch: {epoch}, E_loss: {e_loss:.4f}, G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f}")


def time_gan_trainer(time_gan: TimeGAN, cfg: Dict) -> None:
    # Init all parameters and models
    dataset_name = cfg['system']['dataset']
    sql_len = int(cfg['system']['seq_len'])
    batch_size = int(cfg['system']['batch_size'])
    model_name = cfg['system']['model_name']
    device = cfg['system']['device']
    lr = float(cfg['system']['lr'])

    ds_generator = GeneralDataset.GeneralDataset(sql_len, dataset_name, model_name)
    ds = ds_generator.get_dataset()

    dl = DataLoader(ds, num_workers=10, batch_size=batch_size, shuffle=True)
    time_gan = time_gan.to(device)

    # Optimizers
    # TODO: see the behaviour and update lr with TTsUR if necessary

    emb_opt = Adam(time_gan.emb.parameters(), lr=lr)
    rec_opt = Adam(time_gan.rec.parameters(), lr=lr)
    sup_opt = Adam(time_gan.sup.parameters(), lr=lr)
    g_opt = Adam(time_gan.g.parameters(), lr=lr)
    d_opt = Adam(time_gan.d.parameters(), lr=lr)

    print(f"[EMB] Start Embedding network training")
    embedding_trainer(time_gan=time_gan, emb_opt=emb_opt, rec_opt=rec_opt, dl=dl, cfg=cfg)

    print(f"[SUP] Start Supervisor network training")
    supervisor_trainer(time_gan=time_gan, sup_opt=sup_opt, dl=dl, cfg=cfg)

    print(f"[JOINT] Start joint training")
    joint_trainer(time_gan=time_gan,
                  emb_opt=emb_opt,
                  rec_opt=rec_opt,
                  sup_opt=sup_opt,
                  g_opt=g_opt,
                  d_opt=d_opt,
                  dl=dl,
                  cfg=cfg)


if __name__ == '__main__':
    torch.random.manual_seed(42)

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    run_name = config['system']['run_name'] + ' ' + config['system']['dataset']
    wandb.init(config=config, project='_timegan_baseline_', name=run_name)

    time_gan = TimeGAN(cfg=config)
    time_gan_trainer(time_gan=time_gan, cfg=config)

    # torch.save(time_gan.g.state_dict(), './trained_models/rcgan_g.pt')
    # torch.save(time_gan.d.state_dict(), './trained_models/rcgan_d.pt')
    #
    # time_gan.g = time_gan.g.to(config['system']['device'])
    # time_gan.d = time_gan.d.to(config['system']['device'])

