import torch
from torch import nn
import tinycudann as tcnn
import numpy as np
class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class HashGrid(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # # #big hashgrid from gaussians in the wild
        # desired_resolution = 2048
        # m_base_grid_resolution = 16.0
        # m_n_levels = 12
        # m_per_level_scale = np.exp(np.log(desired_resolution / m_base_grid_resolution) / (m_n_levels - 1))
        # n_features_per_level = 2
        # log2_hashmap_size = 19

        # #small hashgrid from 3D viewdir
        m_base_grid_resolution = 12.0 #16.0
        m_n_levels = 4
        m_per_level_scale = 1.5
        n_features_per_level = 2 #4
        log2_hashmap_size = 15

        hashgrid_params = {
            "otype": "Grid",           #// Component type.
            "type": "Hash",            #// Type of backing storage of the
                                    #// grids. Can be "Hash", "Tiled"
                                    #// or "Dense".
            "n_levels": m_n_levels,            #// Number of levels (resolutions)
            "n_features_per_level": n_features_per_level, #// Dimensionality of feature vector
                                    #// stored in each level's entries.
            "log2_hashmap_size": log2_hashmap_size,   #// If type is "Hash", is the base-2
                                    #// logarithm of the number of elements
                                    #// in each backing hash table.
            "base_resolution": m_base_grid_resolution,     #// The resolution of the coarsest le-
                                    #// vel is base_resolution^input_dims.
            "per_level_scale": m_per_level_scale,    #// The geometric growth factor, i.e.
                                    #// the factor by which the resolution
                                    #// of each grid is larger (per axis)
                                    #// than that of the preceding level.
            "interpolation": "Linear"  #// How to interpolate nearby grid
                                    #// lookups. Can be "Nearest", "Linear",
                                    #// or "Smoothstep" (for smooth deri-
                                    #// vatives).
        }
        self.encoding = tcnn.Encoding(input_dim, hashgrid_params)
    def forward(self, x):
        return self.encoding(x)
        
class MLP(nn.Module):
    def __init__(self, sh_degree, mlp_W, mlp_D, use_hg, encoding_dims):
        """
        """
        super().__init__()
        self.D = mlp_D - 1
        self.W = mlp_W
        self.campose_dims = 3
        self.viewdir_dims = 3
        self.color_dims = 3
        self.distance_dims = 1
        self.light_I_dim = 1
        self.cos_dim = 1
        self.use_hg = use_hg

        self.sh_degree=sh_degree
        self.features_dc_dim = 3*1
        if self.sh_degree==0:
            self.features_rest_dim=0*3
        elif self.sh_degree==1:
            self.features_rest_dim=3*3
        elif self.sh_degree==2:
            self.features_rest_dim=8*3
        elif self.sh_degree==3:
            self.features_rest_dim=15*3
        else:
            raise NotImplemented('sh>3 not implemented')
        
        self.encoding_dims = encoding_dims

        self.inputs_dim = self.distance_dims + self.viewdir_dims*2 +self.cos_dim + self.color_dims
        if self.use_hg:
            self.inputs_dim += self.encoding_dims
                           

        # layers
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.inputs_dim, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"layer_{i+1}", layer)

        self.out = nn.Linear(self.W, 3)


    def forward(self, x):
        """
        """
 
        input = x
        for i in range(self.D):
            input = getattr(self, f"layer_{i+1}")(input)

        outputs = self.out(input)
        return outputs
        

