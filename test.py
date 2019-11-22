#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import math
from tqdm import trange
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, sys
import gym
import gym.envs.box2d
import cv2

import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp

from torchvision import transforms
from collections import deque
from os.path import join, exists
from models import *

from collections import namedtuple
from hparams import HyperParams as hp

from array2gif import write_gif
import PIL

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

device = 'cpu'

transform = transforms.Compose([
    transforms.ToTensor()
])


def obs2tensor(obs):
    binary_road = obs2feature(obs) # (10, 10)
    s = binary_road.flatten()
    s = torch.tensor(s.reshape([1, -1]), dtype=torch.float)
    obs = np.ascontiguousarray(obs)
    obs = transform(obs).unsqueeze(0)
    return obs.to(device), s.to(device)

def obs2feature(s):
    upper_field = s[:84, 6:90] # we crop side of screen as they carry little information
    img = cv2.cvtColor(upper_field, cv2.COLOR_RGB2GRAY)
    upper_field_bw = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    upper_field_bw = cv2.resize(upper_field_bw, (10, 10), interpolation = cv2.INTER_NEAREST) # re scaled to 7x7 pixels
    upper_field_bw = upper_field_bw.astype(np.float32)/255
    return upper_field_bw

def set_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def test_process(global_agent, vae, rnn, update_term, pid, state_dims, hidden_dims, lr, n_play=1, seed=0):
    env = gym.make('CarRacing-v0')
    set_seed(seed, env=env)
    env.verbose = 0
    env.render()
    agent = global_agent
    
    scores = []
    step = 0
    for ep in range(n_play):
        gif = []
        env.reset()
        score = 0.
        i = 0
        next_hidden = [torch.zeros(1, 1, hp.rnn_hunits).to(device) for _ in range(2)]
        for _ in range(5):
            if not record:
                env.render()
            else:
                img = env.render(mode='rgb_array')
                gif.append(img)
            next_obs, reward, done, _ = env.step(agent.possible_actions[-2])
            score += reward
        next_obs, next_s = obs2tensor(next_obs)
        with torch.no_grad():
            next_latent_mu, _ = vae.encoder(next_obs)
        

        while True:
            if not record:
                env.render()
            else:
                img = env.render(mode='rgb_array')
                gif.append(img)

            obs = next_obs
            s = next_s
            hidden = next_hidden
            latent_mu = next_latent_mu

            # Select action about time t
            
            if hp.use_binary_feature:
                state = torch.cat([latent_mu, hidden[0].squeeze(0), s], dim=1)
            else:
                state = torch.cat([latent_mu, hidden[0].squeeze(0)], dim=1)

            action, _ = agent.select_action(state) # nparray, tensor
            next_obs, reward, done, _ = env.step(action.reshape([-1]))

            with torch.no_grad():
                next_obs, next_s = obs2tensor(next_obs)
                next_latent_mu, _ = vae.encoder(next_obs)

            # MDN-RNN about time t+1
            with torch.no_grad():
                action = torch.tensor(action, dtype=torch.float).view(1, -1).to(device)
                vision_action = torch.cat([next_latent_mu, action], dim=-1) #
                vision_action = vision_action.view(1, 1, -1)
                _, _, _, next_hidden = rnn.infer(vision_action, hidden) #

            # next_state = torch.cat([next_latent_mu, next_hidden[0], next_s], dim=1)

            # Scores
            score += reward
            
            if done:
                scores.append(score)
                print('PID: {}, Ep: {}, Replays: {}, Mean: {:.2f}, Score: {:.2f}'.format(pid, ep, len(agent.replay), np.mean(scores), score))
                break

            i += 1
            step += 1
        if score >= np.max(scores):
            best_gif = gif
        pdict = {
            'agent': agent,
            'scores': scores,
            'avgs': np.mean(scores),
            'step': step,
            'n_episodes': ep,
            'seed': seed,
            'update_term': update_term,
        }
    save_ckpt(pdict, 'test', save_model=False)
    np.save('scores.npy', scores)
    env.close()
    return pdict, best_gif

def save_ckpt(info, filename, root='ckpt', add_prefix=None, save_model=True):
    if add_prefix is None:
        ckpt_dir = os.path.join(root, type(info['agent']).__name__)
    else:
        ckpt_dir = os.path.join(root, add_prefix, type(info['agent']).__name__)
    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if save_model:
        torch.save(
            info, '{}/{}.pth.tar'.format(ckpt_dir, filename)
        )
    plt.figure()
    plt.plot(info['scores'])
    plt.plot(info['avgs'])
    plt.savefig('{}/scores-{}.png'.format(ckpt_dir, filename))

def play(n_play, seed, record):
    vae_path = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'vae', '*.pth.tar')))[-1]
    vae_state = torch.load(vae_path, map_location={'cuda:0': str(device)})

    rnn_path = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'rnn', '*.pth.tar')))[-1]
    rnn_state = torch.load(rnn_path, map_location={'cuda:0': str(device)})

    agent_path = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'A3C', '*.pth.tar')))[-1]
    agent_state = torch.load(agent_path, map_location={'cuda:0': str(device)})

    vae = VAE(hp.vsize).to(device)
    vae.load_state_dict(vae_state['model'])
    vae.eval()

    rnn = RNN(hp.vsize, hp.asize, hp.rnn_hunits).to(device)
    rnn.load_state_dict(rnn_state['model'])
    rnn.eval()

    print('Loaded VAE: {}\n RNN: {}\n Agent: {}\n'.format(vae_path, rnn_path, agent_path))

    state_dims = hp.vsize + hp.rnn_hunits + 100 if hp.use_binary_feature else hp.vsize + hp.rnn_hunits
    hidden_dims = hp.ctrl_hidden_dims
    lr = 1e-4

    global_agent = A3C(input_dims=state_dims, hidden_dims=hidden_dims, lr=lr).to(device)
    global_agent.share_memory()
    global_agent.load_state_dict(agent_state['agent'].state_dict())

    _, gif = test_process(global_agent, vae, rnn, 0, 0, state_dims, hidden_dims, lr, n_play, seed, record)
    
    if record:
        gif = list(
            map(lambda img: np.array(PIL.Image.fromarray(img[::2, ::2, :], 'RGB')\
                ).transpose([2,0,1]),
                gif)
        )
        write_gif(gif, 'a3c.gif', fps=30)


if __name__ == '__main__':
    n_play = int(sys.argv[1])
    seed = int(sys.argv[2])
    record = sys.argv[3]
    if record in ['True', '1']:
        record = True
    play(n_play, seed, record)