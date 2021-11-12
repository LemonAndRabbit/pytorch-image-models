import os
import _pickle as pickle
import numpy as np

import torch

from .sparsity_info import SparsityInfo

def convert_tensor_to_sparsity_info(input_tensor: torch.Tensor, zero_threshold=1e-5):
    return torch.abs(input_tensor).gt(zero_threshold)

def write_sparsity_info(input_tensor: torch.Tensor, file_name: str, zero_threshold=1e-5, dims=['channel', 'height', 'width']):
    assert input_tensor.dim()-1 == len(dims), "dims=%d not equal to input_tensor dim=%d" % (len(dims), input_tensor.dim()-1)

    for tensor in input_tensor:
        sparsity_tensor = np.packbits(convert_tensor_to_sparsity_info(tensor, zero_threshold).cpu().numpy())

        #check if res_conv.csv exists and if not create it
        if not os.path.exists(file_name):
            with open(file_name, 'wb') as f:
                pickle.dump(dims, f)
                pickle.dump(input_tensor.shape[1:], f)
        with open(file_name, "ab") as f:
            pickle.dump(sparsity_tensor, f)

def read_sparsity_info(file_name: str):
    '''get sparsity info from file'''
    sparsity_info_list = []

    with open(file_name, 'rb') as f:
        dims = pickle.load(f)
        shape = pickle.load(f)
        while True:
            try:
                temp = np.unpackbits(pickle.load(f)).reshape(shape)
                sparsity_info_list.append(temp)
            except EOFError:
                break
    
    return SparsityInfo(sparsity_info_list, dims)

'''
def sparsity_analyzer(weight_tensor, zero_threshold=1e-5, dim=1):
    weight_tensor = torch.clone(weight_tensor)
    #analyze element_wise sparsity 
    weight_tensor_flatten = torch.clone(weight_tensor)
    weight_tensor_flatten = weight_tensor_flatten.flatten()
    zero_num=0
    zero_num=torch.sum(weight_tensor_flatten <= zero_threshold).float()
    ele_sparsity = zero_num/weight_tensor_flatten.shape[0]
    
    return [ele_sparsity.item(), zero_num.item()]

def group_sparsity_analyzer_2D(weight_tensor, cube_dim, zero_threshold=1e-5, dim=1):
    results = []
    for i in range(max(1, weight_tensor.shape[0]//cube_dim[0])):
        for j in range(max(1,weight_tensor.shape[1]//cube_dim[1])):
                results.append(sparsity_analyzer(weight_tensor[i*cube_dim[0]:(i+1)*cube_dim[0],j*cube_dim[1]:(j+1)*cube_dim[1]],zero_threshold=zero_threshold,dim=dim)
                                + [i, j])
    return results


def group_sparsity_analyzer_3D(weight_tensor, cube_dim, zero_threshold=1e-5, dim=1):
    results = []
    for i in range(max(1, weight_tensor.shape[0]//cube_dim[0])):
        for j in range(max(1,weight_tensor.shape[1]//cube_dim[1])):
            for k in range(max(1,weight_tensor.shape[2]//cube_dim[2])):
                results.append(sparsity_analyzer(weight_tensor[i*cube_dim[0]:(i+1)*cube_dim[0],j*cube_dim[1]:(j+1)*cube_dim[1],k*cube_dim[2]:(k+1)*cube_dim[2]],zero_threshold=zero_threshold,dim=dim)
                                + [i, j, k])
    return results

def group_sparsity_analyzer_4D(weight_tensor, cube_dim, zero_threshold=1e-5, dim=1):
    results = []
    for i in range(max(1,weight_tensor.shape[0]//cube_dim[0])):
        for j in range(max(1,weight_tensor.shape[1]//cube_dim[1])):
            for k in range(max(1,weight_tensor.shape[2]//cube_dim[2])):
                for l in range(max(1,weight_tensor.shape[3]//cube_dim[3])):
                    results.append(sparsity_analyzer(weight_tensor[i*cube_dim[0]:(i+1)*cube_dim[0],j*cube_dim[1]:(j+1)*cube_dim[1],k*cube_dim[2]:(k+1)*cube_dim[2],l*cube_dim[3]:(l+1)*cube_dim[3]],zero_threshold=zero_threshold,dim=dim) 
                                    + [i, j, k, l])
    return results

def generate_4D_sparsity_record(operator_name, input_tensor, tile_size=[1,7,7]):
    sparsity_list = group_sparsity_analyzer_4D(input_tensor.data.cpu(), 
                                                [input_tensor.data.shape[0], tile_size[0], tile_size[1], tile_size[2]])
    
    #check if res_conv.csv exists and if not create it
    if not os.path.exists(operator_name + ".csv"):
        with open(operator_name + ".csv", 'w') as f:
            f.write(str(input_tensor.shape[1]) + ',' + str(input_tensor.shape[2]) + 
                ',' + str(str(input_tensor.shape[3])) + ',\n')
            f.write(', '.join(map(str, tile_size)) + ',\n')
    with open(operator_name + ".csv", "a") as f:
        for sparsity_info in sparsity_list:
            for i in sparsity_info:
                f.write(str(i))
                f.write(",")
            f.write("\n")
        f.write('instance end,\n')
        
def generate_3D_sparsity_record(operator_name, input_tensor, tile_size=[1, 1]):
    sparsity_list = group_sparsity_analyzer_3D(input_tensor.data.cpu(), 
                                                [input_tensor.data.shape[0], tile_size[0], tile_size[1]])
    
    #check if res_conv.csv exists and if not create it
    if not os.path.exists(operator_name + ".csv"):
        with open(operator_name + ".csv", 'w') as f:
            f.write(str(input_tensor.shape[1]) + ',' + str(input_tensor.shape[2]) + ',\n')
            f.write(', '.join(map(str, tile_size)) + ',\n')
    with open(operator_name + ".csv", "a") as f:
        for sparsity_info in sparsity_list:
            for i in sparsity_info:
                f.write(str(i))
                f.write(",")
            f.write("\n")
        f.write('instance end,\n')

def generate_2D_sparsity_record(operator_name, input_tensor, tile_size=[8,]):
    sparsity_list = group_sparsity_analyzer_2D(input_tensor.data.cpu(), 
                                                [input_tensor.data.shape[0], tile_size[0]])
    
    #check if res_conv.csv exists and if not create it
    if not os.path.exists(operator_name + ".csv"):
        with open(operator_name + ".csv", 'w') as f:
            f.write(str(input_tensor.shape[1]) + ',\n')
            f.write(', '.join(map(str, tile_size)) + ',\n')
    with open(operator_name + ".csv", "a") as f:
        for sparsity_info in sparsity_list:
            for i in sparsity_info:
                f.write(str(i))
                f.write(",")
            f.write("\n")
        f.write('instance end,\n')
'''