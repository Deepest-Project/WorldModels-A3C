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

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
logdir = 'logs'

# gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 64, 64

# ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE, FSIZE = 3, 32, 256, 64, 64, 100

MAX_R = 1.

transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])


def obs2tensor(obs):
    binary_road = obs2feature(obs) # (10, 10)
    s = binary_road.flatten()
    s = torch.tensor(s.reshape([1, -1]), dtype=torch.float)
    obs = np.ascontiguousarray(obs)
    # obs = torch.tensor(obs, dtype=torch.float)
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

def test_process(global_agent, vae, rnn, update_term, pid, state_dims, hidden_dims, lr, device=None, seed=0):
    env = gym.make('CarRacing-v0')
    set_seed(seed, env=env)
    env.verbose = 0
    env.render()
    agent = A3C(input_dims=state_dims, hidden_dims=hidden_dims, lr=lr).to(device)
    
    scores = [-100,]
    best_score = hp.save_start_score
    running_means = []
    step = 0
    worse = 0
    best_agent_state = None
    feat_dir = hp.extra_dir
    os.makedirs(feat_dir, exist_ok=True)

    for ep in range(200, 200+test_ep):
        agent.load_state_dict(global_agent.state_dict())
        env.reset()
        score = 0.
        t = 0
        next_hidden = [torch.zeros(1, 1, hp.rnn_hunits).to(device) for _ in range(2)]
        for _ in range(5):
            env.render()
            next_obs, reward, done, _ = env.step(agent.possible_actions[-2])
            score += reward
        next_obs_tensor, next_s = obs2tensor(next_obs)
        with torch.no_grad():
            next_latent_mu, _ = vae.encoder(next_obs_tensor)
        
        obs_lst = []
        action_lst = []
        reward_lst = []
        next_obs_lst = []
        done_lst = []

        while True:
            env.render()
            obs = next_obs
            obs_tensor = next_obs_tensor
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
            np.savez(
                os.path.join(feat_dir, 'rollout_{:03d}_{:04d}'.format(ep, t)),
                obs=obs,
                action=action.reshape([-1]),
                reward=reward,
                next_obs=next_obs,
                done=done,
            )
            
            obs_lst.append(obs)
            action_lst.append(action.reshape([-1]))
            reward_lst.append(reward)
            next_obs_lst.append(next_obs)
            done_lst.append(done)

            with torch.no_grad():
                next_obs_tensor, next_s = obs2tensor(next_obs)
                next_latent_mu, _ = vae.encoder(next_obs_tensor)

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
                running_mean = np.mean(scores[-30:])
                scores.append(score)
                running_means.append(running_mean)
                print('PID: {}, Ep: {}, Replays: {}, Running Mean: {:.2f}, Score: {:.2f}'.format(pid, ep, len(agent.replay), running_mean, score))
                np.savez(
                    os.path.join(feat_dir, 'rollout_ep_{:03d}'.format(ep)),
                    obs=np.stack(obs_lst, axis=0), # (T, C, H, W)
                    action=np.stack(action_lst, axis=0), # (T, a)
                    reward=np.stack(reward_lst, axis=0), # (T, 1)
                    next_obs=np.stack(next_obs_lst, axis=0), # (T, C, H, W)
                    done=np.stack(done_lst, axis=0), # (T, 1)
                )
                break

            t += 1
            step += 1
        pdict = {
            'agent': agent,
            'scores': scores,
            'avgs': running_means,
            'step': step,
            'n_episodes': ep,
            'seed': seed,
            'update_term': update_term,
        }
        if score > best_score:
            best_score = score
    save_ckpt(pdict, 'test', save_model=False)
    env.close()
    return pdict

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


# ### V model & M model

vae_path = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'vae', '*.pth.tar')))[-1]
vae_state = torch.load(vae_path, map_location={'cuda:0': str(device)})

rnn_path = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'rnn', '*.pth.tar')))[-1]
rnn_state = torch.load(rnn_path, map_location={'cuda:0': str(device)})

agent_path = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'A3C', '*.pth.tar')))[-1]
agent_state = torch.load(agent_path, map_location={'cuda:0': str(device)})

vae = VAE(hp.vsize).to(device)
vae.load_state_dict(vae_state['model'])
vae.eval()

# rnn = MDNRNN(hp.vsize, hp.asize, hp.rnn_hunits, hp.n_gaussians).to(device)
rnn = RNN(hp.vsize, hp.asize, hp.rnn_hunits).to(device)
rnn.load_state_dict(rnn_state['model'])
# mdnrnn.load_state_dict({k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})
rnn.eval()

print('Loaded VAE: {}\n RNN: {}\n Agent: {}\n'.format(vae_path, rnn_path, agent_path))

# ###  Environment

total_infos = []
test_ep = 300

state_dims = hp.vsize + hp.rnn_hunits + 100 if hp.use_binary_feature else hp.vsize + hp.rnn_hunits
hidden_dims = 512
lr = 1e-4

global_agent = A3C(input_dims=state_dims, hidden_dims=hidden_dims, lr=lr).to(device)
global_agent.share_memory()
# import pdb; pdb.set_trace()
global_agent.load_state_dict(agent_state['agent'].state_dict())

p = mp.Process(target=test_process, args=(global_agent, vae, rnn, 0, 0, state_dims, hidden_dims, lr,))
p.start()
p.join()

