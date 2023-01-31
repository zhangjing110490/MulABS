from .reader import Reader, FMReader, CTRReader


def get_reader(config):
    if config.model_name == 'test':
        return Reader(config)
    if config.model_name == 'FM':
        return FMReader(config)
    else:
        return CTRReader(config)
