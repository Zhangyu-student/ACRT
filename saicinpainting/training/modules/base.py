import abc
from typing import Tuple, List

import torch
import torch.nn as nn



class BaseDiscriminator(nn.Module):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Predict scores and get intermediate activations. Useful for feature matching loss
        :return tuple (scores, list of intermediate activations)
        """
        raise NotImplemented()


def get_conv_block_ctor(kind='default'):
    if not isinstance(kind, str):
        return kind
    if kind == 'default':
        return nn.Conv2d
    raise ValueError(f'Unknown convolutional block kind {kind}')


def get_norm_layer(kind='bn'):
    if not isinstance(kind, str):
        return kind
    if kind == 'bn':
        return nn.BatchNorm2d
    if kind == 'in':
        return nn.InstanceNorm2d
    raise ValueError(f'Unknown norm block kind {kind}')


def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')
