# -*- coding: UTF-8 -*-
""""
Created on 08.04.20
Unit tests for samplers
:author:     Martin Dočekal
"""
import unittest
from torch.utils.data import Dataset
from windpytorchutils.samplers import IndicesSubsampler


class MackUpDataset(Dataset):
    """
    Mock up dataset for testing.
    """

    def __init__(self, lenOfDataset):
        """
        Initialization of dataset.

        :param lenOfDataset: Number of samples.
        :type lenOfDataset: int
        """

        self.lDataset = lenOfDataset

    def __len__(self):
        return self.lDataset

    def __getitem__(self, item):
        return item


class TestIndicesSubsampler(unittest.TestCase):
    """
    Unit test of the IndicesSubsampler class.
    """

    def test_sampling(self):
        """
        Test the IndicesSubsampler.
        """
        lDataset = 1000
        sampler = IndicesSubsampler(source=MackUpDataset(lDataset), subsetLen=20)

        o = [x for x in sampler]
        self.assertLess(max(o), lDataset)
        self.assertEqual(len(o), 20)
        self.assertEqual(len(set(o)), 20)

if __name__ == '__main__':
    unittest.main()
