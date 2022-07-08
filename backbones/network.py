import copy
from torch import load
import logging
from torchvision.models import resnet18, resnet34, resnet50, resnet101

def get_key(conv_type):
    dst_key = None
    if 'resnet' in conv_type :
        dst_key = "bn1."
    return dst_key

def get_model(backbone_type, pretrained=False, pretrain_path=None, **kwargs):
    name = backbone_type.lower()
    net = None
    if name == 'resnet18':
        logging.info('created resnet18!')
        net = resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        net = resnet34(pretrained=pretrained)
        logging.info('created resnet34!')
    elif name == 'resnet50':
        net = resnet50(pretrained=pretrained)
        logging.info('created resnet50!')
    elif name == 'resnet101':
        net = resnet101(pretrained=pretrained)
        logging.info('created resnet101!')
    else:
        raise NotImplementedError('Unknown type {}'.format(backbone_type))
    
    # 载入自定义预训练模型
    if pretrain_path != None and pretrained:
        dst_key = get_key(name)

        state_dict=net.state_dict()
        logging.info("{}weight before update: {}".format(dst_key, net.state_dict()[dst_key + "weight"][:5]))
        logging.info("{}bias before update: {}".format(dst_key, net.state_dict()[dst_key + "bias"][:5]))
        logging.info(' ')
        
        pretrained_dict = load(pretrain_path)['state_dict']
        state_dict = net.state_dict()
        logging.info('special keys in load model state dict: {}'.format(pretrained_dict.keys()-state_dict.keys()))
        for key in (pretrained_dict.keys() & state_dict.keys()):
            state_dict[key] = pretrained_dict[key]
        net.load_state_dict(state_dict)

        logging.info("loaded pretrained_dict_name: {}".format(pretrain_path))
        logging.info("{}weight after update: {}".format(dst_key, net.state_dict()[dst_key + "weight"][:5]))
        logging.info("{}bias after update: {}".format(dst_key, net.state_dict()[dst_key + "bias"][:5]))
        logging.info(' ')
    
    return net