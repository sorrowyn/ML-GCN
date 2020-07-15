from .gcn import GCNResnet

def build_model(config, num_classes, concur, sums, pretrained=True, use_gpu=True):
    dict_paramsters = None
    if config['name'] == 'gcn_resnet':
        dict_paramsters = {'backbone': config['backbone'],
                            'in_channel': config['encode_dim'],
                            'threshold': config['threshold']}
        model = GCNResnet(num_classes, config['backbone'], concur, sums, config['encode_dim'], config['threshold'])
    else:
        raise KeyError('config[model][name] error')
    return model, dict_paramsters