from backbones.bal_sup_moco_fc_netV2 import BalSupMoCoFCNet
from tqdm import tqdm
import torchvision.models as models
from torch import nn
import logging
from torch.nn.functional import softmax
from utils.toolkit import plot_confusion_matrix, plot_ROC_curve
from methods.mocov2 import MoCoV2
import torch
from os.path import join
import numpy as np

class BalSupMoCo_FC(MoCoV2):
    def __init__(self, args, seed):
        super().__init__(args, seed)
        self.network = BalSupMoCoFCNet(models.__dict__[args['backbone']], K=args['K'], m=args['m'], T=args['T'], mlp=True)
        self.gamma = args['gamma']

        self.network = self.network.cuda()
        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)
    
    def epoch_train(self, dataloader, optimizer, scheduler):
        statistic_ce = 0.
        statistic_contrast = 0.

        self.network.train()
        losses = 0.
        with tqdm(total=len(dataloader), ncols=150) as prog_bar:
            for inputs, targets in dataloader:
                inputs[0], inputs[1], targets = inputs[0].cuda(), inputs[1].cuda(), targets.cuda()
                contrast_loss = self.network.comput_supcontrast_loss(inputs[0], inputs[1], targets).mean()
                ce_loss = self.criterion(self.network(inputs[0]), targets)
                loss = (1-self.gamma)*contrast_loss + self.gamma*ce_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                statistic_ce += ce_loss.item()
                statistic_contrast += contrast_loss.item()

                prog_bar.update(1)

        scheduler.step()

        train_loss = losses/len(dataloader)
        logging.info('ce loss: {} , contrastive loss: {} , all: {}'.format(statistic_ce/len(dataloader),statistic_contrast/len(dataloader), train_loss))

        return None, train_loss

    def after_train(self, dataloaders, tblog=None):
        # valid
        if not dataloaders['valid'] is None:
            all_preds, all_labels, all_scores = self.get_output(dataloaders['valid'])

            # ??? confusion matrix ??? ROC curve ????????? tensorboard
            cm_name = self.method+'_'+self.backbone+"_valid_Confusion_Matrix"
            cm_figure, tp, fp, fn, tn = plot_confusion_matrix(all_labels, all_preds, ['non-BA', 'BA'], cm_name)
            if tblog is not None:
                tblog.add_figure(cm_name, cm_figure)

            acc = torch.sum(all_preds == all_labels).item() / len(all_labels)
            recall = tn / (tn + fp)
            precision = tn / (tn + fn)
            specificity = tp / (tp + fn)
            logging.info('===== Evaluate valid set result ======')
            logging.info('acc = {:.4f} precision = {:.4f} , recall = {:.4f} , specificity = {:.4f}'.format(acc, precision, recall, specificity))

        # test        
        all_preds, all_labels, all_scores = self.get_output(dataloaders['test'])

        # ??? confusion matrix ??? ROC curve ????????? tensorboard
        cm_name = self.method+'_'+self.backbone+"_test_Confusion_Matrix"
        cm_figure, tp, fp, fn, tn = plot_confusion_matrix(all_labels, all_preds, ['non-BA', 'BA'], cm_name)
        if tblog is not None:
            tblog.add_figure(cm_name, cm_figure)

        roc_name = self.method+'_'+self.backbone+"_test_ROC_Curve"
        roc_auc, roc_figure, opt_threshold, opt_point = plot_ROC_curve(all_labels, all_scores, ['non-BA', 'BA'], roc_name)
        if tblog is not None:
            tblog.add_figure(roc_name, roc_figure)

        # ?????? precision ??? recall??? ??? zero_division ??????0????????? precision ???0????????????warning
        acc = torch.sum(all_preds == all_labels).item() / len(all_labels)
        recall = tn / (tn + fp)
        precision = tn / (tn + fn)
        specificity = tp / (tp + fn)

        logging.info('===== Evaluate test set result ======')
        logging.info('acc = {:.4f} , auc = {:.4f} , precision = {:.4f} , recall = {:.4f} , specificity = {:.4f}'.format(acc, roc_auc, precision, recall, specificity))

    def get_output(self, dataloader):
        self.network.eval()

        all_preds = torch.tensor([])
        all_labels = torch.tensor([])
        all_scores = torch.tensor([])

        with torch.no_grad():
            with tqdm(total=len(dataloader), ncols=150) as _tqdm:   # ???????????????
                for inputs, labels in dataloader:
                    inputs = inputs.cuda()
                    outputs = self.network(inputs)
                    _, preds = torch.max(outputs, 1)
                    scores = softmax(outputs.detach().cpu(),1)
                    preds = preds.detach().cpu()
                    all_preds = torch.cat((all_preds.long(), preds.long()), 0)
                    all_labels = torch.cat((all_labels.long(), labels), 0)
                    all_scores = torch.cat((all_scores,scores), 0)

                    _tqdm.update(1)
        
        return all_preds, all_labels, all_scores

    
    def save_checkpoint(self, filename, model=None, state_dict=None):
        save_path = join(self.save_dir, filename+'.pkl')
        # if state_dict != None:
        #     save_dict = state_dict
        # else:
        #     save_dict = model.encoder_q.state_dict()
        torch.save(model, save_path)
        logging.info('model state dict saved at: {}'.format(save_path))


        
        
        
        

        

