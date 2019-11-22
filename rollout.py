import numpy as np
import os, sys, glob
import gym
from hparams import HyperParams as hp

def rollout(agent):
    env = gym.make("CarRacing-v0")

    seq_len = 1000
    max_ep = hp.n_rollout
    feat_dir = hp.data_dir

    os.makedirs(feat_dir, exist_ok=True)

    for ep in range(max_ep):
        obs_lst, action_lst, reward_lst, next_obs_lst, done_lst = [], [], [], [], []
        env.reset()
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        done = False
        t = 0
        
        while not done or t < seq_len:
            t += 1

            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)

            np.savez(
                os.path.join(feat_dir, 'rollout_{:03d}_{:04d}'.format(ep,t)),
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )
            
            obs_lst.append(obs)
            action_lst.append(action)
            reward_lst.append(reward)
            next_obs_lst.append(next_obs)
            done_lst.append(done)
            obs = next_obs
        np.savez(
            os.path.join(feat_dir, 'rollout_ep_{:03d}'.format(ep)),
            obs=np.stack(obs_lst, axis=0), # (T, C, H, W)
            action=np.stack(action_lst, axis=0), # (T, a)
            reward=np.stack(reward_lst, axis=0), # (T, 1)
            next_obs=np.stack(next_obs_lst, axis=0), # (T, C, H, W)
            done=np.stack(done_lst, axis=0), # (T, 1)
        )
        
        

if __name__ == '__main__':
    agent = sys.argv[1] # random or pretrained
    np.random.seed(hp.seed)
    rollout(agent)
