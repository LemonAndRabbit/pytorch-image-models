from math import ceil
from functools import reduce

import numpy as np


def tile_analyzer(tile_tensor: np.ndarray):
    tile_tensor_flatten = tile_tensor.flatten().astype(int)
    zero_num = np.sum(tile_tensor_flatten)
    
    return zero_num

def tile_2D(input_tensor: np.ndarray, tile_size: list):
    result_shape = [0, 0]
    for i in range(2):
        result_shape[i] = ceil(input_tensor.shape[i] / tile_size[i])
    result_tensor = np.zeros(result_shape, dtype=float)

    for i in range(result_shape[0]):
        for j in range(result_shape[1]):
            result_tensor[i, j] = tile_analyzer(input_tensor[
                i*tile_size[0]:min(input_tensor.shape[0], (i+1)*tile_size[0]),
                j*tile_size[1]:min(input_tensor.shape[1], (j+1)*tile_size[1])])
    
    return result_tensor / reduce(lambda a,b: a*b, tile_size)

def tile_3D(input_tensor: np.ndarray, tile_size: list):
    result_shape = [0, 0, 0]
    for i in range(3):
        result_shape[i] = ceil(input_tensor.shape[i] / tile_size[i])
    result_tensor = np.zeros(result_shape, dtype=float)

    for i in range(result_shape[0]):
        for j in range(result_shape[1]):
            for k in range(result_shape[2]):
                result_tensor[i, j] = tile_analyzer(input_tensor[
                    i*tile_size[0]:min(input_tensor.shape[0], (i+1)*tile_size[0]),
                    j*tile_size[1]:min(input_tensor.shape[1], (j+1)*tile_size[1]),
                    k*tile_size[2]:min(input_tensor.shape[2], (k+1)*tile_size[2])])

    return result_tensor / reduce(lambda a,b: a*b, tile_size)