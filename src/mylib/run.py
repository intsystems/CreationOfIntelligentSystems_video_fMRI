import torch
from data_preprocess_and_load.dataset import fMRIDataset
from torch.utils.data import DataLoader

def main():
    args = {'lr_init':1e-3, 'lr_gamma':0.97, 'lr_step':1000,
        'weight_decay':1e-7, 'task':'transformer_reconstruction', 'cuda':False, 
        'transformer_hidden_layers':2, 'batch_size':8,
        'reconstruction_factor':1, 'perceptual_factor':1, 'intensity_factor':1,
        'nEpochs':5, 'memory_constraint':0.1, 'loaded_model_weights_path':'AutoEncoder_last_epoch.pth'}

    dataset = fMRIDataset(seq_len=5)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, drop_last=True)

    trainer = Trainer(dataloader, **args)
    trainer.training()

if __name__ == '__main__':
    main()
