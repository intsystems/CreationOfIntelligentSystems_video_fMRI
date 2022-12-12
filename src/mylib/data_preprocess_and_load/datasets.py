import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from torchvision.transforms import ToTensor


class fMRIDataset(Dataset):
    def __init__(self, seq_len: int = 1, total_len=641):
        super().__init__()
        self.data_dir = os.path.join('ds003688-download', 'fMRI_to_tensors')
        self.subjects_data = os.listdir(self.data_dir)
        self.total_len = total_len
        self.seq_len = seq_len
    
    def __getitem__(self, index):
        subj_dir = self.subjects_data[index % len(self.subjects_data)]
        start_idx = index // len(self.subjects_data)

        glob_tensors = torch.stack([torch.load(os.path.join(self.data_dir, subj_dir, 'glob_norm', f'rfMRI__TR_{start_idx + i}.pt'))
            for i in range(self.seq_len)], dim=0)

        vox_tensors = torch.stack([torch.load(os.path.join(self.data_dir, subj_dir, 'vox_norm', f'rfMRI__TR_{start_idx + i}.pt'))
            for i in range(self.seq_len)], dim=0)
        
        glob_tensors = glob_tensors.transpose(0, -1)
        vox_tensors = vox_tensors.transpose(0, -1)
        return torch.cat([glob_tensors, vox_tensors], dim=0)
    
    def __len__(self):
        return len(self.subjects_data) * (self.total_len - self.seq_len + 1)


class fMRIVideoDataset(Dataset):
    def __init__(self, train=True, seq_len=1, video_path='Film stimulus.mp4', skip_frames=1, total_len=641):
        super().__init__()
        assert seq_len % 5 == 0
        self.seq_len = seq_len
        self.total_len = total_len
        self.num_chunks = seq_len // 5
        self.skip_frames = skip_frames

        self.data_dir = os.path.join('ds003688-download', 'fMRI_to_tensors')
        subjects_data = sorted(os.listdir(self.data_dir))
        subjects_data_train, subjects_data_test = train_test_split(subjects_data, test_size=0.3, random_state=0)
        self.start_times = torch.arange(5, self.total_len - 5*self.num_chunks + 1, 5)
        if train:
            self.subjects_data = subjects_data_train
        else:
            self.subjects_data = subjects_data_test

        self.get_frames(video_path)

    def get_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        to_tensor = ToTensor()
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224,224),fx=0,fy=0, interpolation=cv2.INTER_CUBIC)
                frames.append(to_tensor(frame))
            else:
                break
        # When everything done, release the video capture object
        cap.release()
        self.frames = frames


    def __getitem__(self, index):
        subj_dir = self.subjects_data[index % len(self.subjects_data)]
        start_idx = self.start_times[index // len(self.subjects_data)]
        start_frame = (start_idx // 5) * 76
        end_frame = start_frame + self.num_chunks * 76
        video = torch.stack(self.frames[start_frame:end_frame:self.skip_frames])
        position_idx = torch.arange(0, end_frame - start_frame, self.skip_frames)

        glob_tensors = torch.stack([torch.load(os.path.join(self.data_dir, subj_dir, 'glob_norm', f'rfMRI__TR_{start_idx + i}.pt'))
            for i in range(self.seq_len)], dim=0)

        vox_tensors = torch.stack([torch.load(os.path.join(self.data_dir, subj_dir, 'vox_norm', f'rfMRI__TR_{start_idx + i}.pt'))
            for i in range(self.seq_len)], dim=0)
        
        glob_tensors = glob_tensors.transpose(0, -1)
        vox_tensors = vox_tensors.transpose(0, -1)
        TFF_input = torch.cat([glob_tensors, vox_tensors], dim=0)

        prev_glob_tensor = torch.load(os.path.join(self.data_dir, subj_dir, 'glob_norm', f'rfMRI__TR_{start_idx - 1}.pt'))
        prev_vox_tensor = torch.load(os.path.join(self.data_dir, subj_dir, 'vox_norm', f'rfMRI__TR_{start_idx - 1}.pt'))
        prev_tensor = torch.stack([prev_glob_tensor, prev_vox_tensor], dim=0)

        return {'fmri_seq':TFF_input, 'fmri_img':prev_tensor, 'video_seq':video, 'pos_idx':position_idx}
    
    def __len__(self):
        return len(self.subjects_data) * len(self.start_times)
