import torch
import torch.nn as nn
import numpy as np
from hparams import RNNHyperParams as hp
from models import VAE, RNN
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

    # Loaded pretrained VAE
    vae = VAE(hp.vsize).to(DEVICE)
    ckpt = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'vae', '*k.pth.tar')))[-1]
    vae_state = torch.load(ckpt)
    vae.load_state_dict(vae_state['model'])
    vae.eval()
    print('Loaded vae ckpt {}'.format(ckpt))

    rnn = RNN(hp.vsize, hp.asize, hp.rnn_hunits).to(DEVICE)
    ckpts = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'rnn', '*k.pth.tar')))
    if ckpts:
        ckpt = ckpts[-1]
        rnn_state = torch.load(ckpt)
        rnn.load_state_dict(rnn_state['model'])
        global_step = int(os.path.basename(ckpt).split('.')[0][:-1]) * 1000
        print('Loaded rnn ckpt {}'.format(ckpt))


    data_path = hp.data_dir if not hp.extra else hp.extra_dir
    # optimizer = torch.optim.RMSprop(rnn.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)
    dataset = GameEpisodeDataset(data_path, seq_len=hp.seq_len)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, drop_last=True,
        num_workers=hp.n_workers, collate_fn=collate_fn
    )
    testset = GameEpisodeDataset(data_path, seq_len=hp.seq_len, training=False)
    test_loader = DataLoader(
        testset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn
    )

    ckpt_dir = os.path.join(hp.ckpt_dir, 'rnn')
    sample_dir = os.path.join(ckpt_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    l1 = nn.L1Loss()
    
    while global_step < hp.max_step:
        # GO_states = torch.zeros([hp.batch_size, 1, hp.vsize+hp.asize]).to(DEVICE)
        with tqdm(enumerate(loader), total=len(loader), ncols=70, leave=False) as t:
            t.set_description('Step {}'.format(global_step))
            for idx, (obs, actions) in t:
                obs, actions = obs.to(DEVICE), actions.to(DEVICE)
                with torch.no_grad():
                    latent_mu, latent_var = vae.encoder(obs) # (B*T, vsize)
                    z = latent_mu
                    # z = vae.reparam(latent_mu, latent_var) # (B*T, vsize)
                    z = z.view(-1, hp.seq_len, hp.vsize) # (B*n_seq, T, vsize)
                # import pdb; pdb.set_trace()
                
                next_z = z[:, 1:, :]
                z, actions = z[:, :-1, :], actions[:, :-1, :]
                states = torch.cat([z, actions], dim=-1) # (B, T, vsize+asize)
                # states = torch.cat([GO_states, next_states[:,:-1,:]], dim=1)
                x, _, _ = rnn(states)
                
                loss = l1(x, next_z)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1

                if global_step % hp.log_interval == 0:
                    eval_loss = evaluate(test_loader, vae, rnn, global_step)
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(os.path.join(ckpt_dir, 'train.log'), 'a') as f:
                        log = '{} || Step: {}, train_loss: {:.4f}, loss: {:.4f}\n'.format(now, global_step, loss.item(), eval_loss)
                        f.write(log)
                    S = 2
                    y = vae.decoder(x[S, :, :])
                    v = vae.decoder(next_z[S, :, :])
                    save_image(y, os.path.join(sample_dir, '{:04d}-rnn.png'.format(global_step)))
                    save_image(v, os.path.join(sample_dir, '{:04d}-vae.png'.format(global_step)))
                    save_image(obs[S:S+hp.seq_len-1], os.path.join(sample_dir, '{:04d}-obs.png'.format(global_step)))
                
                if global_step % hp.save_interval == 0:
                    d = {
                        'model': rnn.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(
                        d, os.path.join(ckpt_dir, '{:03d}k.pth.tar'.format(global_step//1000))
                    )

def evaluate(test_loader, vae, rnn, global_step=0):
    rnn.eval()
    total_loss = []
    l1 = nn.L1Loss()
    with torch.no_grad():
        for idx, (obs, actions) in enumerate(test_loader):
            obs, actions = obs.to(DEVICE), actions.to(DEVICE)
            latent_mu, latent_var = vae.encoder(obs) # (B*T, vsize)
            z = latent_mu
            # z = vae.reparam(latent_mu, latent_var) # (B*T, vsize)
            z = z.view(-1, hp.seq_len, hp.vsize) # (B*n_seq, T, vsize)

            next_z = z[:, 1:, :]
            z, actions = z[:, :-1, :], actions[:, :-1, :]
            states = torch.cat([z, actions], dim=-1) # (B, T, vsize+asize)
            # states = torch.cat([GO_states, next_states[:,:-1,:]], dim=1)
            x, _, _ = rnn(states)
            
            loss = l1(x, next_z)
 
            total_loss.append(loss.item())
    rnn.train()
    return np.mean(total_loss)


if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(hp.seed)
    train()