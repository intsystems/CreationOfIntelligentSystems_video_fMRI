import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class fMRIDataset(Dataset):
    def __init__(self, seq_len: int = 1):
        super().__init__()
        self.data_dir = os.path.join('ds003688-download', 'fMRI_to_tensors')
        self.subjects_data = os.listdir(self.data_dir)
        self.seq_len = seq_len
    
    def __getitem__(self, index):
        subj_dir = self.subjects_data[index]
        total_len = len(os.listdir(os.path.join(self.data_dir, subj_dir, 'glob_norm')))
        start_idx = np.random.choice(total_len - self.seq_len + 1)
        glob_tensors = torch.stack([torch.load(os.path.join(self.data_dir, subj_dir, 'glob_norm', f'rfMRI__TR_{start_idx + i}.pt'))
            for i in range(self.seq_len)], dim=0)

        vox_tensors = torch.stack([torch.load(os.path.join(self.data_dir, subj_dir, 'vox_norm', f'rfMRI__TR_{start_idx + i}.pt'))
            for i in range(self.seq_len)], dim=0)
        
        glob_tensors = glob_tensors.transpose(0, -1)
        vox_tensors = vox_tensors.transpose(0, -1)
        return torch.cat([glob_tensors, vox_tensors], dim=0)
    
    def __len__(self):
        return len(self.subjects_data)
