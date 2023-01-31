from .configs import FMConfig, CTRConfig, RFConfig, LRConfig


def get_config(arch):
    if arch == 'RF':
        return RFConfig()
    elif arch == 'LR':
        return LRConfig()
    elif arch == 'FM':
        return FMConfig()
    elif arch == 'DeepFM' or arch == 'CTR':
        return CTRConfig(arch)
    else:
        raise ValueError(f'{arch} is not supported. Only CTR, FM, LR, DeepFM, RF are supported')
