from data_preprocess_and_load.datasets import *
import numpy as np
import torch
import argparse
from trainer import Trainer
import os
from pathlib import Path
from collections import OrderedDict


def get_arguments(base_path):
    """
    handle arguments from commandline.
    some other hyper parameters can only be changed manually (such as model architecture,dropout,etc)
    notice some arguments are global and take effect for the entire three phase training process, while others are determined per phase
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--directory', type=str, default=os.path.join(base_path, 'TFF_weights'))
    parser.add_argument('--intensity_factor', default=1)
    parser.add_argument('--perceptual_factor', default=1)
    parser.add_argument('--reconstruction_factor', default=1)
    parser.add_argument('--KL_factor', default=1e-3)
    parser.add_argument('--transformer_hidden_layers', default=2)

    ##phase 1
    parser.add_argument('--task_phase1', type=str, default='autoencoder_reconstruction')
    parser.add_argument('--batch_size_phase1', type=int, default=16)
    parser.add_argument('--nEpochs_phase1', type=int, default=5)
    parser.add_argument('--weight_decay_phase1', default=1e-7)
    parser.add_argument('--lr_init_phase1', default=1e-3)
    parser.add_argument('--lr_gamma_phase1', default=0.97)
    parser.add_argument('--lr_step_phase1', default=500)
    parser.add_argument('--seq_len_phase1', default=1)
    parser.add_argument('--memory_constraint_phase1', default=0.1)
    parser.add_argument('--title_phase1', default='AutoEncoder')

    ##phase 2
    parser.add_argument('--task_phase2', type=str, default='transformer_reconstruction')
    parser.add_argument('--batch_size_phase2', type=int, default=8)
    parser.add_argument('--nEpochs_phase2', type=int, default=5)
    parser.add_argument('--weight_decay_phase2', default=1e-7)
    parser.add_argument('--lr_gamma_phase2', default=0.97)
    parser.add_argument('--lr_step_phase2', default=1000)
    parser.add_argument('--seq_len_phase2', default=5)
    parser.add_argument('--memory_constraint_phase2', default=0.1)
    parser.add_argument('--title_phase2', default='Transformer')

    ##phase 3
    parser.add_argument('--task_phase3', type=str, default='video_vae')
    parser.add_argument('--batch_size_phase3', type=int, default=1)
    parser.add_argument('--nEpochs_phase3', type=int, default=5)
    parser.add_argument('--weight_decay_phase3', default=1e-7)
    parser.add_argument('--lr_gamma_phase3', default=0.97)
    parser.add_argument('--lr_step_phase3', default=1000)
    parser.add_argument('--seq_len_phase3', default=5)
    parser.add_argument('--memory_constraint_phase3', default=0.1)
    parser.add_argument('--title_phase3', default='VAE')
    parser.add_argument('--video_path_phase3', default='Film stimulus.mp4')
    parser.add_argument('--skip_frames_phase3', default=4)
    parser.add_argument('--vtn_path_phase3', default='VTN_VIT_B_KINETICS.pyth')
    
    args = parser.parse_args()
    return args
  
def sort_args(phase, args):
    phase_specific_args = {}
    for name, value in args.items():
        if not 'phase' in name:
            phase_specific_args[name] = value
        elif 'phase' + phase in name:
            phase_specific_args[name.replace('_phase' + phase, '')] = value
    return phase_specific_args

def run_phase(args,loaded_model_weights_path,phase_num,phase_name):
    """
    main process that runs each training phase
    :return path to model weights (pytorch file .pth) aquried by the current training phase
    """
    
    setattr(args,'loaded_model_weights_path_phase' + phase_num, loaded_model_weights_path)
    args = sort_args(phase_num, vars(args))
    trainer = Trainer(**args)
    trainer.training()
    return

def main(base_path):
    args = get_arguments(base_path)
    # pretrain step1
    print('starting phase 1...')
    run_phase(args,None,'1','autoencoder_reconstruction')
    print('finishing phase 1...')
    #pretrain step2
    print('starting phase 2...')
    model_weights_path_phase1 = os.path.join(base_path, 'TFF_weights/AutoEncoder_last_epoch.pth')
    run_phase(args,model_weights_path_phase1, '2', 'tranformer_reconstruction')
    print('finishing phase 2...')
    #fine tune
    print('starting phase 3...')
    model_weights_path_phase2 = os.path.join(base_path, 'TFF_weights/Transformer_last_epoch.pth')
    run_phase(args, model_weights_path_phase2,'3','video_vae')
    print('finishing phase 3...')


if __name__ == '__main__':
    base_path = os.getcwd()
    os.makedirs(os.path.join(base_path,'TFF_weights'),exist_ok=True)
    main(base_path)
