import torch
import torch.nn as nn
import numpy as np
from hparams import VAEHyperParams as hp
from models import VAE, vae_loss
from torch.utils.data import DataLoader
from data import *
from tqdm import tqdm
import os, sys
from torchvision.utils import save_image
from torch.nn import functional as F
from datetime import datetime

DEVICE = None


def train():
    global_step = 0
    model = VAE(hp.vsize).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Loaded pretrained VAE
    ckpts = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'vae', '*k.pth.tar')))
    if ckpts:
        ckpt = ckpts[-1]
        vae_state = torch.load(ckpt)
        model.load_state_dict(vae_state['model'])
        global_step = int(os.path.basename(ckpt).split('.')[0][:-1]) * 1000
        print('Loaded vae ckpt {}'.format(ckpt))

    data_path = hp.data_dir if not hp.extra else hp.extra_dir
    dataset = GameSceneDataset(data_path)
    loader = DataLoader(
        dataset, batch_size=hp.batch_size, shuffle=True,
        num_workers=hp.n_workers,
    )
    testset = GameSceneDataset(data_path, training=False)
    test_loader = DataLoader(testset, batch_size=hp.test_batch, shuffle=False, drop_last=True)

    ckpt_dir = os.path.join(hp.ckpt_dir, 'vae')
    sample_dir = os.path.join(ckpt_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    while global_step < hp.max_step:
        for idx, obs in enumerate(tqdm(loader, total=len(loader))):
            x = obs.to(DEVICE)
            x_hat, mu, logvar = model(x)
            
            loss, recon_loss, kld = vae_loss(x_hat, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % hp.log_interval == 0:
                recon_loss, kld = evaluate(test_loader, model, sample_dir, global_step)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(os.path.join(ckpt_dir, 'train.log'), 'a') as f:
                    log = '{} || Step: {}, loss: {:.4f}, kld: {:.4f}\n'.format(now, global_step, recon_loss, kld)
                    f.write(log)
            
            if global_step % hp.save_interval == 0:
                d = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(
                    d, os.path.join(ckpt_dir, '{:03d}k.pth.tar'.format(global_step//1000))
                )
            global_step += 1

def evaluate(test_loader, model, sample_dir=None, global_step=0):
    model.eval()
    total_recon_loss = []
    total_kld_loss = []
    n_sample = hp.n_sample
    c_x = torch.zeros([n_sample, 3, 96, 96])
    c_x_hat = torch.zeros([n_sample, 3, 96, 96])
    with torch.no_grad():
        for idx, obs in enumerate(test_loader):
            x = obs.to(DEVICE)
            # import pdb; pdb.set_trace()
            x_hat, mu, logvar = model(x)
            _, recon_loss, kld = vae_loss(x_hat, x, mu, logvar)

            if idx < n_sample:
                c_x[idx] = x[0]
                c_x_hat[idx] = x_hat[0]
            total_recon_loss.append(recon_loss.item())
            total_kld_loss.append(kld.item())
        z = torch.randn([n_sample, hp.vsize]).to(DEVICE)
        x_rand = model.decoder(z)
    save_image(x_rand, os.path.join(sample_dir, '{:04d}k-random.png'.format(global_step//1000)))
    save_image(c_x_hat, os.path.join(sample_dir, '{:04d}k-xhat.png'.format(global_step//1000)))
    save_image(c_x, os.path.join(sample_dir, '{:04d}k-x.png'.format(global_step//1000)))
    model.train()
    return np.mean(total_recon_loss), np.mean(total_kld_loss)


if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    train()