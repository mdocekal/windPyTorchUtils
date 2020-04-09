# -*- coding: UTF-8 -*-
""""
Created on 08.04.20
Module with PyTorch samplers.

:author:     Martin Dočekal
"""
import torch
from torch.utils.data import Sampler, Dataset


class IndicesSubsampler(Sampler):
    """
    Sample for subsampling.
        https://en.wikipedia.org/wiki/Resampling_(statistics)#Subsampling

    These sampler does not provides the data itself, but just the indices that should be selected.
    """

    def __init__(self, source: Dataset, subsetLen: int):
        """
        Len of subsampled dataset.

        :param source: Source dataset you want to sample from. We need it just for the len.
        :type source: Dataset
        :param subsetLen: Len of dataset after subsampling.
        :type subsetLen: int
        """
        super().__init__()
        assert subsetLen <= len(source)

        self.source = source
        self.subsetLen = subsetLen

    def __len__(self):
        return self.subsetLen

    def __iter__(self):
        for x in torch.randperm(len(self.source))[:self.subsetLen]:
            yield int(x)