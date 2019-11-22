import torch
import torch.nn as nn
import numpy as np
import glob, os
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class GameSceneDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, training=True, test_ratio=0.01):
        self.fpaths = sorted(glob.glob(os.path.join(data_path, 'rollout_[0-9][0-9][0-9]_*.npz')))
        np.random.seed(0)
        indices = np.arange(0, len(self.fpaths))
        n_trainset = int(len(indices)*(1.0-test_ratio))
        self.train_indices = indices[:n_trainset]
        self.test_indices = indices[n_trainset:]
        # self.train_indices = np.random.choice(indices, int(len(indices)*(1.0-test_ratio)), replace=False)
        # self.test_indices = np.delete(indices, self.train_indices)
        self.indices = self.train_indices if training else self.test_indices
        # import pdb; pdb.set_trace()

    def __getitem__(self, idx):
        npz = np.load(self.fpaths[self.indices[idx]])
        obs = npz['obs']
        obs = transform(obs)
        # obs = obs.permute(2, 0, 1) # (N, C, H, W)
        return obs

    def __len__(self):
        return len(self.indices)

class GameEpisodeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len=32, seq_mode=True, training=True, test_ratio=0.01):
        self.training = training
        self.fpaths = sorted(glob.glob(os.path.join(data_path, 'rollout_ep_*.npz')))
        np.random.seed(0)
        indices = np.arange(0, len(self.fpaths))
        n_trainset = int(len(indices)*(1.0-test_ratio))
        self.train_indices = indices[:n_trainset]
        self.test_indices = indices[n_trainset:]
        # self.train_indices = np.random.choice(indices, int(len(indices)*(1.0-test_ratio)), replace=False)
        # self.test_indices = np.delete(indices, self.train_indices)
        self.indices = self.train_indices if training else self.test_indices
        self.seq_len = seq_len
        self.seq_mode = seq_mode
        # import pdb; pdb.set_trace()

    def __getitem__(self, idx):
        npz = np.load(self.fpaths[self.indices[idx]])
        obs = npz['obs'] # (T, H, W, C) np array
        actions = npz['action'] # (T, n_actions) np array
        T, H, W, C = obs.shape
        n_seq = T // self.seq_len
        end_seq = n_seq * self.seq_len # T' = end of sequence
        
        obs = obs[:end_seq].reshape([-1, self.seq_len, H, W, C]) # (N_seq, seq_len, H, W, C)
        actions = actions[:end_seq].reshape([-1, self.seq_len, actions.shape[-1]]) # 

        # if args.seq_mode:
        #     start_range = max_len-self.seq_len
        #     for t in range(0, max_len-self.seq_len, self.seq_len):
        #         obs[t:t+self.seq_len]
        # else:
        #     rand_start = np.random.randint(max_len-self.seq_len)
        #     obs = obs[rand_start:rand_start+self.seq_len] # (T, H, W, C)
        #     actions = actions[rand_start:rand_start+self.seq_len]
        return obs, actions

    def __len__(self):
        return len(self.indices)

def collate_fn(data):
    # obs (B, N_seq, seq_len, H, W, C), actions (B, N_seq, seq_len, n_actions)
    obs, actions = zip(*data)
    obs, actions = np.array(obs), np.array(actions)
    _, _, seq_len, H, W, C = obs.shape
    obs = obs.reshape([-1, H, W, C]) # (B*N_seq*seq_len, H, W, C)
    actions = actions.reshape([-1, seq_len, actions.shape[-1]]) # (B*n_seq, n_actions)
    obs_lst = []
    for i in range(len(obs)): # batch loop
        obs_lst.append(transform(obs[i]))
        # for j in range(len(obs[i])): # sequence loop
        #     obs_lst.append(transform(obs[i][j]))
    obs = torch.stack(obs_lst, dim=0) # (B*N_seq*seq_len, C, H, W)
    # obs = obs.view([-1, seq_len, H, W, C]) # (B*N_seq, seq_len, C, H, W)
    return obs, torch.tensor(actions, dtype=torch.float)