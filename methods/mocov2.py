from torch.nn import functional as F
from backbones.moco_net import MoCoNet
from methods.base import Base
import torchvision.models as models
from torch import nn
from tqdm import tqdm
from os.path import join
import logging
import torch

class MoCoV2(Base):
    def __init__(self, args, seed):
        super().__init__(args, seed)
        self.network = MoCoNet(models.__dict__[args['backbone']], K=args['K'], m=args['m'], T=args['T'], mlp=True)
        # if args['freeze']:
        #     self.network.freeze()
        self.network = self.network.cuda()
        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)

    def epoch_train(self, dataloader, optimizer, scheduler):
        self.network.train()
        losses = 0.
        with tqdm(total=len(dataloader), ncols=150) as prog_bar:
            for inputs, targets in dataloader:
                inputs[0], inputs[1], targets = inputs[0].cuda(), inputs[1].cuda(), targets.cuda()
                logits, labels = self.network(inputs[0], inputs[1], targets)

                loss = F.cross_entropy(logits, labels) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            
                prog_bar.update(1)

        scheduler.step()

        train_loss = losses/len(dataloader)
        return None, train_loss
    
    def compute_accuracy(self, model, loader):
        return None

    def save_checkpoint(self, filename, model=None, state_dict=None):
        save_path = join(self.save_dir, filename+'.pkl')
        if state_dict != None:
            save_dict = state_dict
        else:
            save_dict = model.encoder_q.state_dict()
        torch.save({
            'state_dict': save_dict,
            'backbone': self.backbone
            }, save_path)
        logging.info('model state dict saved at: {}'.format(save_path))


        
        
        
        

        

