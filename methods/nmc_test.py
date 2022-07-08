from numpy import isin
from torch import nn
from methods.base import Base
import logging
from backbones.network import get_model
import torch
from tqdm import tqdm
from torch.nn.functional import softmax
from utils.toolkit import plot_confusion_matrix, plot_ROC_curve
from utils.toolkit import count_parameters

class NMC(Base):
    def __init__(self, args, seed):
        super().__init__(args, seed)
        self.backbone = args['backbone']
        self.network = get_model(args['backbone'], args['pretrained'], args['pretrain_path'])

        self.class_means = []

        for name, param in self.network.named_parameters():
            param.requires_grad = False
            logging.info("{} require grad={}".format(name, param.requires_grad))

        self.network = self.network.cuda()
        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)
    
    def train_model(self, dataloaders, tblog, valid_epoch=1):
        logging.info('All params before training: {}'.format(count_parameters(self.network)))
        logging.info('Trainable params: {}'.format(count_parameters(self.network, True)))
        # for name, param in self.network.named_parameters():
        #     logging.info("{} require grad={}".format(name, param.requires_grad))
        self.cal_class_means(dataloaders['train'])
    
    def after_train(self, dataloaders, tblog=None):
        # test        
        test_features = self.get_features(dataloaders['test'])
        all_preds, all_labels = self.get_output(dataloaders['test'])

        # 将 confusion matrix 和 ROC curve 输出到 tensorboard
        cm_name = self.method+'_'+self.backbone+"_test_Confusion_Matrix"
        cm_figure, tp, fp, fn, tn = plot_confusion_matrix(all_labels, all_preds, ['non-BA', 'BA'], cm_name)
        if tblog is not None:
            tblog.add_figure(cm_name, cm_figure)


        # 计算 precision 和 recall， 将 zero_division 置为0，使当 precision 为0时不出现warning
        acc = torch.sum(all_preds == all_labels).item() / len(all_labels)
        recall = tn / (tn + fp)
        precision = tn / (tn + fn)
        specificity = tp / (tp + fn)

        logging.info('===== Evaluate test set result ======')
        logging.info('acc = {:.4f} , auc = {:.4f} , precision = {:.4f} , recall = {:.4f} , specificity = {:.4f}'.format(acc, roc_auc, precision, recall, specificity))

        if self.save_models:
            self.save_checkpoint(self.network.cpu(), 'model_dict')

    def get_features(self, dataloader):
        self.network.eval()

        all_labels = torch.tensor([])
        all_features = torch.tensor([])

        with torch.no_grad():
            with tqdm(total=len(dataloader), ncols=150) as _tqdm:   # 显示进度条
                for inputs, labels in dataloader:
                    inputs = inputs.cuda()
                    outputs = self.network(inputs)
                    all_features = torch.cat((all_features, outputs), 0)
                    all_labels = torch.cat((all_labels.long(), labels), 0)

                    _tqdm.update(1)
        
        return all_features, all_labels

    
    def cal_class_means(self, dataloader):
        class_means = []
        all_features, all_labels = self.get_features(dataloader)
        class_labels = torch.unique(all_labels)
        for i in range(class_labels.shape[0]):
            class_ids = all_labels[torch.where(all_labels==class_labels[i])[0]]
            class_means.append(all_features[class_ids].mean(dim=1).unsqueeze(0))
        self.class_means = torch.cat(class_means)


        
        

            
