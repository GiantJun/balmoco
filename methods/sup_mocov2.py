from backbones.sup_moco_net import SupMoCoNet
from tqdm import tqdm
import torchvision.models as models
from torch import nn
from os.path import join
from methods.mocov2 import MoCoV2

class SupMoCoV2(MoCoV2):
    def __init__(self, args, seed):
        super().__init__(args, seed)
        self.network = SupMoCoNet(models.__dict__[args['backbone']], K=args['K'], m=args['m'], T=args['T'], mlp=True)

        self.network = self.network.cuda()
        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)
    
    def epoch_train(self, dataloader, optimizer, scheduler):
        self.network.train()
        losses = 0.
        with tqdm(total=len(dataloader), ncols=150) as prog_bar:
            for inputs, targets in dataloader:
                inputs[0], inputs[1], targets = inputs[0].cuda(), inputs[1].cuda(), targets.cuda()
                loss = self.network(inputs[0], inputs[1], targets).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                prog_bar.update(1)

        scheduler.step()

        train_loss = losses/len(dataloader)
        return None, train_loss



        
        
        
        

        

