from .ctr import CTR
from .fm import FM
from .deepfm import DeepFM
from .rf import RF
from .lr import LR
import torch


def get_model(config, embed_features, embedding_dict, n_labels, device):
    if config.model_name == 'FM':
        model = FM(config, embed_features, embedding_dict, n_labels, device)
    elif config.model_name == 'CTR':
        model = CTR(config, embed_features, embedding_dict, device)
    elif config.model_name == 'DeepFM':
        model = DeepFM(config, embed_features, embedding_dict, device)
    elif config.model_name == 'RF':
        model = RF(config, embed_features)
    elif config.model_name == 'LR':
        model = LR(config, embed_features)
    else:
        raise ValueError(f'{config.model_name} is not supported. Only CTR, FM, LR, DeepFM, RF are supported')

    if config.load_model:
        model.load_state_dict(torch.load(config.save_path))

    return model
