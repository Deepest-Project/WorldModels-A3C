import torch
import torch.nn as nn
import numpy as np
from hparams import RNNHyperParams as hp
from models import *
from torch.utils.data import DataLoader
from data import *
from tqdm import tqdm
import os, sys
from torchvision.utils import save_image
from torch.nn import functional as F
from datetime import datetime

DEVICE = None

def dream():
    # Loaded pretrained VAE
    vae = VAE(hp.vsize).to(DEVICE)
    ckpt = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'vae', '*k.pth')))[-1]
    vae_state = torch.load(ckpt)
    vae.load_state_dict(vae_state['model'])
    vae.eval()
    print('Loaded vae ckpt {}'.format(ckpt))

    if hp.memory == 'MDNRNN':
        rnn = MDNRNN(hp.vsize, hp.asize, hp.rnn_hunits, hp.n_gaussians).to(DEVICE)
    else:
        rnn = RNN(hp.vsize, hp.asize, hp.rnn_hunits).to(DEVICE)
    ckpt = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'rnn', '*k.pth')))[-1]
    rnn_state = torch.load(ckpt)
    rnn.load_state_dict(rnn_state['model'])
    rnn.eval()
    print('Loaded RNN ckpt {}'.format(ckpt))
    data_path = hp.data_dir if not hp.extra else hp.extra_dir

    testset = GameEpisodeDataset(data_path, seq_len=hp.seq_len, training=False)
    test_loader = DataLoader(
        testset, batch_size=hp.test_batch, shuffle=False, drop_last=False, collate_fn=collate_fn
    )

    ckpt_dir = os.path.join(hp.ckpt_dir, 'rnn')
    sample_dir = os.path.join(ckpt_dir, 'dreams')
    os.makedirs(sample_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, (obs, actions) in enumerate(tqdm(test_loader, total=len(test_loader), ncols=70)):
            obs, actions = obs.to(DEVICE), actions.to(DEVICE)
        
            latent_mu, latent_var = vae.encoder(obs) # (B*n_seq*T, vsize)
            z = latent_mu
            # z = vae.reparam(latent_mu, latent_var) # (B*T, vsize)
            z = z.view(hp.batch_size, -1, hp.vsize) # (B, T, vsize)
            actions = actions.view(hp.batch_size, -1, hp.asize) # (B, T, vsize)
            
            # next_z = z[:, 1:, :]
            # z, actions = z[:, :-1, :], actions[:, :-1, :]
            # states = torch.cat([z, actions], dim=-1) # (B, T, vsize+asize)

            
            c = 20 # starting point
            simul_term = 128
            # hidden = [torch.zeros([1,1,hp.rnn_hunits]).to(DEVICE) for _ in range(2)]
            while c < 1000 - simul_term:
                xs = []
                zc = z[0:1, c, :] # (1, vsize)
                zc = torch.randn_like(zc)
                hidden = [torch.zeros([1,1,hp.rnn_hunits]).to(DEVICE) for _ in range(2)]
                for t in range(c, c+simul_term):
                    zc = zc.unsqueeze(1) # (1, 1, vsize)
                    
                    # action = actions[0:1, t:t+1, :] # (1, 1, asize)
                    action = torch.tensor([[[0., 0.5, 0.0]]]).to(DEVICE)

                    state = torch.cat([zc, action], dim=-1) # (1, 1, vsize+asize)
                    # mu, _, pi, hidden = rnn.infer(state, hidden)
                    mu, hidden, _, _ = rnn.infer(state, hidden)
                    # pi.max()

                    x = vae.decoder(mu[:, 0, :]) # (1, C, H, W)
                    xs.append(x)

                    zc, _ = vae.encoder(x) # (1, vsize)

                    # v = vae.decoder(next_z[t, :, :])
                xs = torch.cat(xs, dim=0) # (T, C, H, W)
                save_image(xs[::2], os.path.join(sample_dir, '{}-{}.png'.format(idx,c)))
                # save_image(v, os.path.join(sample_dir, '{}-vae.png'.format(t)))
                obs = obs.view([hp.batch_size, -1, hp.img_channels, hp.img_height, hp.img_width])
                save_image(obs[0, c:c+simul_term], os.path.join(sample_dir, '{}-{}-obs.png'.format(idx,c)))
                c = c + simul_term


if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = 'cpu'
    np.random.seed(hp.seed)
    dream()