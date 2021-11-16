from copy import deepcopy

import numpy as np
import matplotlib

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


cdict = {'red':   [[0.0,  1.0, 1.0],
                   [0.6,  0.5, 0.0],
                   [1.0,  0.0, 0.0]],
         'green': [[0.0,  0.0, 0.0],
                   [1.0,  0.0, 0.0]],
         'blue':  [[0.0,  0.0, 0.0],
                   [0.6,  0.0, 0.5],
                   [1.0,  1.0, 1.0]]}
cmap = LinearSegmentedColormap('cmap', segmentdata=cdict, N=256)

class SparsityMap3D:
    def __init__(self, sparsity_tensor, labels=['channel', 'height', 'width']):
        self.sparsity_tensor = deepcopy(sparsity_tensor)
        self.map_size = self.sparsity_tensor.shape
        self.labels = labels
    
    def draw(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        xs = np.arange(0, self.map_size[0]).repeat(self.map_size[1]*self.map_size[2])
        ys = np.tile(np.arange(0, self.map_size[1]).repeat(self.map_size[2]), self.map_size[0])
        zs = np.tile(np.arange(0, self.map_size[2]), self.map_size[0]*self.map_size[1])
        sparsities = self.sparsity_tensor.flatten()
                
        ax.scatter(xs, ys, zs, c=sparsities, cmap=cmap, norm=Normalize(vmin=0, vmax=1), marker='.')

        ax.set_xlabel(self.labels[0])
        ax.set_ylabel(self.labels[1])
        ax.set_zlabel(self.labels[2])

        fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1)))
        fig.show()
        return fig

class SparsityMap2D:
    def __init__(self, sparsity_tensor, labels=['token', 'channel']):
        self.sparsity_tensor = deepcopy(sparsity_tensor)
        self.map_size = self.sparsity_tensor.shape
        self.labels = labels

    def draw(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        
        xs = np.arange(0, self.map_size[0]).repeat(self.map_size[1])
        ys = np.tile(np.arange(0, self.map_size[1]), self.map_size[0])
        sparsities = self.sparsity_tensor.flatten()
        
        ax.scatter(xs, ys, c=sparsities, cmap=cmap, norm=Normalize(vmin=0, vmax=1), marker='.')

        ax.set_xlabel(self.labels[0])
        ax.set_ylabel(self.labels[1])

        fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1)))
        fig.show()
        return fig

class SparsityMap1D:
    def __init__(self, sparsity_tensor, labels=['channel',]):
        self.sparsity_tensor = deepcopy(sparsity_tensor).flatten()
        self.map_size = self.sparsity_tensor.shape
        self.labels = labels

    def draw_distribution(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        
        ax.hist(self.sparsity_tensor, bins=np.arange(21)*0.05)

        ax.set_ylabel(self.labels[0] + 'count')
        ax.set_xlabel('sparsity')

        fig.show()
    
    def draw(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        
        ax.bar(range(len(self.sparsity_tensor)), self.sparsity_tensor)

        ax.set_xlabel(self.labels[0])
        ax.set_ylabel('sparsity')

        fig.show()
        return fig

