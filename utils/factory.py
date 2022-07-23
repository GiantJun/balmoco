from methods.finetune import Finetune
from methods.mocov2 import MoCoV2
from methods.sup_mocov2 import SupMoCoV2
from methods.balsup_mocov2 import BalSupMoCoV2
from methods.test_model import TestModel
from methods.balsup_mocov2_fc import BalSupMoCo_FC

def get_trainer(args, seed):
    name = args['method'].lower()
    if name == 'finetune':
        return Finetune(args, seed)
    elif name == 'mocov2':
        return MoCoV2(args, seed)
    elif name == 'sup_mocov2':
        return SupMoCoV2(args, seed)
    elif name == 'balsup_mocov2':
        return BalSupMoCoV2(args, seed)
    elif name == 'balsup_mocov2_fc':
        return BalSupMoCo_FC(args, seed)
    elif name == 'test':
        return TestModel(args, seed)
    else:
        raise NotImplementedError('Unknown method {}'.format(name))

