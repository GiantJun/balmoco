import argparse
from utils.toolkit import set_logger
import yaml
from dataset.data_manager import get_dataloader
import torch
from utils.factory import get_trainer
import os
import logging

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='',
                        help='yaml file of settings.')
    return parser

def load_yaml(settings_path):
    args = {}
    with open(settings_path) as data_file:
        param = yaml.load(data_file, Loader=yaml.FullLoader)
    args.update(param['basic'])
    # args.update(param['special'])
    dataset = args['dataset']
    backbone = args['backbone']
    if 'options' in param:
        args.update(param['options'][dataset][backbone])
    if 'special' in param:
        args.update(param['special'])
    return args

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args, seed):
    # log hyperparameter
    logging.info(30*"=")
    logging.info("log hyperparameters in seed {}".format(seed))
    logging.info(30*"-")
    # args = dict(sorted(args.items()))
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
    logging.info(30*"=")

if __name__ == '__main__':
    args = setup_parser().parse_args()
    param = load_yaml(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    if not 'pretrain_path' in args:
        args['pretrain_path'] = None

    os.environ['CUDA_VISIBLE_DEVICES']=args['device']

    tblog = set_logger(args)
    try:
        # 准备数据集
        data_loaders, class_num = get_dataloader(args['dataset'], batch_size=args['batch_size'], num_workers=args['num_workers'], 
                        img_size=args['img_size'], ret_valid=args['ret_valid'])
        args.update({'class_num':class_num})
        
        for seed in args['seed']:
            print_args(args, seed)
            set_random(seed)
            trainer = get_trainer(args, seed)
            trainer.train_model(data_loaders, tblog)
            trainer.after_train(data_loaders, tblog)
    except Exception as e:
        logging.error(e, exc_info=True, stack_info=True)



    
    


    