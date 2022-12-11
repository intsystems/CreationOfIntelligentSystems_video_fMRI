from learning_rate import LrHandler
#from data_preprocess_and_load.dataloaders import DataHandler
import torch
import warnings
from tqdm import tqdm
from model import Encoder_Transformer_Decoder,Encoder_Transformer_finetune,AutoEncoder
from losses import get_intense_voxels

class Trainer():
    """
    main class to handle training, validation and testing.
    note: the order of commands in the constructor is necessary
    """
    def __init__(self, train_dataloader, **kwargs):
        self.register_args(**kwargs)
        self.lr_handler = LrHandler(**kwargs)
        self.train_loader = train_dataloader
        self.create_model()
        self.initialize_weights(load_cls_embedding=False)
        self.create_optimizer()
        self.lr_handler.set_schedule(self.optimizer)

        #self.initialize_weights(load_cls_embedding=False)
        #self.writer = Writer(sets,**kwargs)
        #self.sets = sets
        #self.eval_iter = 0
        self.losses = {'intensity':
                           {'is_active':True,'factor':self.intensity_factor},
                       'perceptual':
                           {'is_active':True, 'factor':self.perceptual_factor},
                       'reconstruction':
                           {'is_active':True,'factor':self.reconstruction_factor}}
        self.reconstruction_loss_func = nn.L1Loss()
        self.perceptual_loss_func = Percept_Loss(self.memory_constraint)
        self.intensity_loss_func = nn.L1Loss() #(thresholds=[0.9, 0.99]

    def initialize_weights(self,load_cls_embedding):
        if self.loaded_model_weights_path is not None:
            state_dict = torch.load(self.loaded_model_weights_path)
            #self.lr_handler.set_lr(state_dict['lr'])
            self.model.load_partial_state_dict(state_dict['model_state_dict'],load_cls_embedding)
            #self.model.loaded_model_weights_path = self.loaded_model_weights_path


    def create_optimizer(self):
        lr = self.lr_handler.base_lr
        params = self.model.parameters()
        weight_decay = self.kwargs.get('weight_decay')
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def create_model(self):
        dim = (40, 64, 64)
        if self.task.lower() == 'fine_tune':
            self.model = Encoder_Transformer_finetune(dim,**self.kwargs)
        elif self.task.lower() == 'autoencoder_reconstruction':
            self.model = AutoEncoder(dim,**self.kwargs)
        elif self.task.lower() == 'transformer_reconstruction':
            self.model = Encoder_Transformer_Decoder(dim,**self.kwargs)
        if self.cuda:
            self.model = self.model.cuda()

    def training(self):
        self.loss_history = {'intensity': [], 'perceptual':[], 'reconstruction':[]}
        for epoch in range(self.nEpochs):
            int_loss, perc_loss, rec_loss = self.train_epoch(epoch)
            self.loss_history['intensity'].append(int_loss)
            self.loss_history['perceptual'].append(perc_loss)
            self.loss_history['reconstruction'].append(rec_loss)

            print('______epoch summary {}/{}_____\n'.format(epoch+1,self.nEpochs))
            print(f'intensity loss {int_loss}, perceptual loss {perc_loss}, reconstruction loss {rec_loss}')
        return self.loss_history


    def train_epoch(self,epoch):
        self.train()
        int_loss = 0
        perc_loss = 0
        rec_loss = 0

        for batch_idx, input_dict in enumerate(tqdm(self.train_loader)):
           # self.writer.total_train_steps += 1
            self.optimizer.zero_grad()
            loss_dict, loss = self.forward_pass(input_dict)
            
            int_loss += loss_dict['intensity']
            perc_loss += loss_dict['perceptual']
            rec_loss += loss_dict['reconstruction']

            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            loss.backward()
            self.optimizer.step()
            self.lr_handler.schedule_check_and_update()

            #self.writer.write_losses(loss_dict, set='train')
        return int_loss / len(self.train_loader), perc_loss / len(self.train_loader), rec_loss / len(self.train_loader)

    def train(self):
        self.mode = 'train'
        self.model = self.model.train()

   # def get_last_loss(self):
   #     if self.model.task == 'regression':
   #         return self.writer.val_MAE[-1]
   #     else:
    #        return self.writer.total_val_loss_history[-1]

    # def get_last_accuracy(self):
    #     if hasattr(self.writer,'val_AUROC'):
    #         return self.writer.val_AUROC[-1]
    #     else:
    #         return None

    # def save_checkpoint_(self,epoch):
    #     loss = self.get_last_loss()
    #     accuracy = self.get_last_accuracy()
    #     self.model.save_checkpoint(
    #         self.writer.experiment_folder, self.writer.experiment_title, epoch, loss ,accuracy, self.optimizer ,schedule=self.lr_handler.schedule)


    def forward_pass(self,input_dict):
        #input_dict = {k:(v.cuda() if self.cuda else v) for k,v in input_dict.items()}
        if self.cuda:
          input_dict = input_dict.cuda()
        output_dict = self.model(input_dict)
        loss_dict, loss = self.aggregate_losses(input_dict, output_dict)

        return loss_dict, loss


    def aggregate_losses(self,input_dict,output_dict):
        final_loss_dict = {}
        final_loss_value = 0
        for loss_name, current_loss_dict in self.losses.items():
            loss_func = getattr(self, 'compute_' + loss_name)
            current_loss_value = loss_func(input_dict,output_dict)
            if current_loss_value.isnan().sum() > 0:
                warnings.warn('found nans in computation')
                print('at {} loss'.format(loss_name))
            lamda = current_loss_dict['factor']
            factored_loss = current_loss_value * lamda
            final_loss_dict[loss_name] = factored_loss.item()
            final_loss_value += factored_loss

        final_loss_dict['total'] = final_loss_value.item()
        return final_loss_dict, final_loss_value

    def compute_reconstruction(self,input_dict,output_dict):
        fmri_sequence = input_dict[:,0].unsqueeze(1)
        reconstruction_loss = self.reconstruction_loss_func(output_dict['reconstructed_fmri_sequence'],fmri_sequence)
        return reconstruction_loss

    def compute_intensity(self,input_dict,output_dict):
        per_voxel = input_dict[:,1,:,:,:,:]
        voxels = get_intense_voxels(per_voxel, output_dict['reconstructed_fmri_sequence'].shape)
        output_intense = output_dict['reconstructed_fmri_sequence'][voxels]
        truth_intense = input_dict[:,0][voxels.squeeze(1)]
        intensity_loss = self.intensity_loss_func(output_intense.squeeze(), truth_intense)
        return intensity_loss

    def compute_perceptual(self,input_dict,output_dict):
        fmri_sequence = input_dict[:,0].unsqueeze(1)
        perceptual_loss = self.perceptual_loss_func(output_dict['reconstructed_fmri_sequence'],fmri_sequence)
        return perceptual_loss

    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs
