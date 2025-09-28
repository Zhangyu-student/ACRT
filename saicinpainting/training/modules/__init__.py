import logging
from archs.S1_arch import DiffIRS1
from saicinpainting.training.modules.pix2pixhd import NLayerDiscriminator

def make_generator(config, kind, **kwargs):
    logging.info(f'Make generator {kind}')
    
    if kind == 'DiffIRS1':
        return DiffIRS1(**kwargs)

    raise ValueError(f'Unknown generator kind {kind}')


def make_discriminator(kind, **kwargs):
    logging.info(f'Make discriminator {kind}')

    if kind == 'pix2pixhd_nlayer':
        return NLayerDiscriminator(**kwargs)

    raise ValueError(f'Unknown discriminator kind {kind}')
