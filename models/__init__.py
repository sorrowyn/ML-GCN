from .gcn import GCNResnet
from .util import gen_A

def build_model(config, num_classes, adjacent_matrix, inp, pretrained=True, use_gpu=True):
    dict_paramsters = None
    if config['name'] == 'gcn_resnet':
        dict_paramsters = {'backbone': config['backbone'],
                            'in_channel': config['encode_dim'],
                            'threshold': config['threshold']}
        model = GCNResnet(num_classes, config['backbone'], adjacent_matrix, inp, config['encode_dim'])
    else:
        raise KeyError('config[model][name] error')
    return model, dict_paramsters