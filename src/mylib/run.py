import torch
from data_preprocess_and_load.datasets import fMRIDataset
from torch.utils.data import DataLoader

from trainer import Trainer
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_init", default=1e-3, type=float, help="initial lerning rate")
    parser.add_argument("--lr_gamma", default=0.97, type=float)
    parser.add_argument("--lr_step", default=1000, type=int)
    parser.add_argument("--weight_decay", default=1e-7, type=float)
    parser.add_argument("--task", default='transformer_reconstruction', type=str)
    parser.add_argument("--cuda", default=False, type=bool)
    parser.add_argument("--transformer_hidden_layers", default=2, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--reconstruction_factor", default=1, type=float)
    parser.add_argument("--perceptual_factor", default=1, type=float)
    parser.add_argument("--intensity_factor", default=1, type=float)
    parser.add_argument("--nEpochs", default=5, type=int)
    parser.add_argument("--memory_constraint", default=0.1, type=float)
    parser.add_argument("--loaded_model_weights_path", default="AutoEncoder_last_epoch.pth", type=str)
    args = parser.parse_args()
    return args

def main(args):
    train_dataset = fMRIDataset(seq_len=5)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    trainer = Trainer(dataloader, **args)
    trainer.training()

if __name__ == '__main__':
    args = get_args()
    main(args)
