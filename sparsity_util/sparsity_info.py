from functools import cached_property
from .utils import tile_2D, tile_3D
from copy import deepcopy

import numpy as np

class SparsityInfo:
    '''store sparsity info in each sample round as a list'''

    def __init__(self, sparsirty_info_list: list, dims: list, tile_size=[]):
        self.batch_count = len(sparsirty_info_list)
        self.sparsity_info_list = deepcopy(sparsirty_info_list)
        self.shape = self.sparsirty_info_list[0].shape
        self.dims = dims
        self.tile_size = tile_size
    
    @cached_property
    def tiled(self):
        return len(self.tile_size) == 0

    @cached_property
    def avg(self):
        '''avgerage over batches'''

        temp = np.zeros(self.shape, dtype=float)  
        if self.tiled:
            for tensor in self.sparsity_info_list:
                temp = temp + tensor.astype(int)
        else:
            for tensor in self.sparsity_info_list:
                temp = temp + tensor

        return temp / self.batch_count

    @cached_property
    def std_var(self):
        '''standard variation over batches'''

        temp = np.zeros(self.shape, dtype=float)
        if self.tiled:
            for tensor in self.sparsity_info_list:
                temp = temp + np.square(tensor.astype(int) - self.avg)
        else:
            for tensor in self.sparsity_info_list:
                temp = temp + np.square(tensor - self.avg)
        
        temp = temp / (self.batch_count - 1)
        return np.sqrt(temp)

    def tile(self, tile_size: list):
        '''create tiled sparsity info'''

        assert self.tiled, "already tiled, retiling not allowed"
        
        assert len(tile_size) == len(self.shape), \
            "tile_size dim=%d not equal to tensor dim=%d" \
            % (len(tile_size), len(self.shape))
        
        output_tensor_list = []
        if len(tile_size) == 2:
            for tensor in self.sparsity_info_list:
                output_tensor_list.append(tile_2D(tensor))
        if len(tile_size) == 3:
            for tensor in self.sparsity_info_list:
                output_tensor_list.append(tile_3D(tensor))

        return SparsityInfo(output_tensor_list, self.dims, tile_size)           

