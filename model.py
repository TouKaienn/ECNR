import torch
import torch.nn as nn
import math
import numpy as np
import torchsnooper
from utils import *
from tools import *
from einops import rearrange, repeat
from icecream import ic
#*-----------------Basic Modules-----------------*#
class PosEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class Sine(nn.Module):
    def __init(self):
        super(Sine,self).__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

#*-----------------Net-----------------*#
class ECNR(nn.Module):
    """
    Fitting siggnal with lots of small MLPs
    """
    def __init__(self,n_MLPs,in_features,n_blocks_perMLP,timeCodeDim,out_features,init_features,n_layers,omega_0=30):
        super(ECNR, self).__init__()
        self.LatentTable = torch.nn.Parameter(torch.randn(n_MLPs,n_blocks_perMLP,timeCodeDim)) # (t, timeCodeDim)
        self.n_layers = n_layers
        self.parameters_to_prune = {}
        self.parameters_to_quant = {}
        self.bias_to_prune = {}
        net = []
        for i in range(n_layers):
            if i == 0:
                net.append(AdaptiveMultiSineLayer(in_features+timeCodeDim,init_features,n_MLPs,is_first=True,omega_0=omega_0))
            elif i < n_layers - 1:
                net.append(AdaptiveMultiSineLayer(init_features,init_features,n_MLPs,omega_0=omega_0))
            else: #final layer
                final_layer = DummyLinearWithChannel(init_features, out_features, n_MLPs)
                net.append(final_layer)
        self.net = nn.ModuleList(net)
        # init prune and quantization variables
        self.weight_masks = {self.net[l].linear:None for l in range(n_layers)} #  for weight sharing used to reconstrcut weight
        self.bias_masks = {self.net[l].linear:None for l in range(n_layers)} #  for bias sharing used to reconstrcut bias
        self.latent_centroids = None
        self.latent_labels = None
        self.is_quanted = False
        self.is_bias_quanted = False
        self.is_latent_quanted = False

    def _set_parameters_to_prune(self,parameters_to_prune,bias_to_prune):
        self.parameters_to_prune = [module for (module,name) in parameters_to_prune]
        self.bias_to_prune = [module for (module,name) in bias_to_prune]
        
        
    def _set_parameters_to_quant(self,parameters_to_quantize,bias_to_quant):
        self.parameters_to_quant = [module for (module,name) in parameters_to_quantize]
        self.bias_to_quant = [module for (module,name) in bias_to_quant]
        
    def apply_weight_share(self,n_bits=8,Global_QUANT=True):
        #* need to remove prune mask before apply weight share
        if not self.is_quanted: # just make sure it is first time run weight share
            self.is_quanted = True
            n_clusters = int(2**n_bits)
            named_buffer_idx = 0
            for l in range(self.n_layers):
                module = self.net[l].linear
                if Global_QUANT:
                    if (self.net[l].linear in self.parameters_to_quant):
                        if (self.net[l].linear in self.parameters_to_prune): # if prune
                            self.weight_masks[module]=torch.FloatTensor(dict(self.named_buffers())[f'net.{l}.linear.weight_mask']).long().to(self.net[0].linear.weight.device) # get prune mask
                            weight = self.net[l].linear.weight.data
                            weight_nonzero = weight.view(-1)[self.weight_masks[module].view(-1)!=0] # get nonzero weight
                            labels,centers = kmeans_quantization(weight_nonzero,n_clusters)
                            self.net[l].apply_weight_share(centers,labels,_weight_mask=self.weight_masks[module])
                            named_buffer_idx += 1
                        else: # if not prune
                            weight = self.net[l].linear.weight.data
                            labels,centers = kmeans_quantization(weight.view(-1),n_clusters)
                            self.net[l].apply_weight_share(centers,labels)
                else:
                    if (self.net[l].linear in self.parameters_to_prune): # if prune
                            self.weight_masks[module]=torch.FloatTensor(dict(self.named_buffers())[f'net.{l}.linear.weight_mask']).long().to(self.net[0].linear.weight.device) # get prune mask
                            weight = self.net[l].linear.weight.data
                            labels = torch.zeros(weight.shape[0],weight.shape[1]*weight.shape[2],dtype=torch.long)
                            centers = torch.zeros(weight.shape[0],n_clusters)
                            weight_nonzero = weight*(self.weight_masks[module]!=0) # weight non_zero
                            # ic(weight_nonzero.shape) # [216,11,23]
                            # labels,centers = multikmeans_quantization(weight_nonzero,n_clusters)
                            for i in range(weight.shape[0]):
                                # weight_nonzero = weight[i,...].view(-1)[self.weight_masks[module][i,...].view(-1)!=0] # get nonzero weight
                                labels[i,:],centers[i,:] = kmeans_quantization(weight_nonzero[i,...].view(-1),n_clusters)
                            
             
                            self.net[l].apply_local_weight_share(centers,labels,_weight_mask=self.weight_masks[module])
                            named_buffer_idx += 1
                            # exit()
                    else: # if not prune
                        weight = self.net[l].linear.weight.data
                        labels,centers = multikmeans_quantization(weight.view(-1),n_clusters)
                        self.net[l].apply_local_weight_share(centers,labels)
                        
    
    def apply_latent_share(self,n_bits=9):
        if not self.is_latent_quanted:
            self.is_latent_quanted = True
            n_clusters = int(2**n_bits)
            latent_weight = self.LatentTable.data
            labels,centroids = kmeans_quantization(latent_weight.view(-1),n_clusters)
            self.latent_centroids = torch.nn.Parameter(centroids).to(self.net[0].linear.weight.device)
            self.latent_labels = torch.nn.Parameter(labels,requires_grad=False).to(self.net[0].linear.weight.device)
            
    def apply_bias_share(self,n_bits=8):
        #* need to remove prune mask before apply weight share
        if not self.is_bias_quanted: # just make sure it is first time run weight share
            self.is_bias_quanted = True
            n_clusters = int(2**n_bits)
            named_buffer_idx = 0
            for l in range(self.n_layers):
                module = self.net[l].linear
                if (self.net[l].linear in self.bias_to_quant):
                    if (self.net[l].linear in self.bias_to_prune): # if prune
                        self.bias_masks[module]=torch.FloatTensor(dict(self.named_buffers())[f'net.{l}.linear.bias_mask']).long().to(self.net[0].linear.weight.device) # get prune mask
                        bias = self.net[l].linear.bias.data           
                        bias_nonzero = bias.view(-1)[self.bias_masks[module].view(-1)!=0] # get nonzero weight
                        labels,centers = kmeans_quantization(bias_nonzero,n_clusters)
                        self.net[l].apply_bias_share(centers,labels,_bias_mask=self.bias_masks[module])
                        named_buffer_idx += 1
                    else: # if not prune
                        bias = self.net[l].linear.bias.data
                        labels,centers = kmeans_quantization(bias.view(-1),n_clusters)
                        self.net[l].apply_bias_share(centers,labels)

    def forward(self,coords,indices,t_indices):
        #coords --> (n_MLPs, batch_size, in_features)
        #indices --> (n_MLPs) Long Tensor <- this term is actually useless, because apply indices to select MLP will not speed up
        #t_indices --> (n_MLPs, batch_size) Long Tensor
        if self.is_latent_quanted:
            LatentTable = torch.zeros_like(self.LatentTable)
            LatentTable.view(-1)[:] = self.latent_centroids[self.latent_labels]
        else:
            LatentTable = self.LatentTable
        LatentCode = LatentTable[:,t_indices[0],:] #(n_MLPs, batch_size, timeCodeDim)
        LatentCode = LatentCode[indices, ...]
        output = coords[indices, ...]
        output = torch.cat([output,LatentCode],dim=-1)
        for l_idx,mod in enumerate(self.net):
            output = mod(output, indices)
        return output
    
    def inf(self,coords,indices,label):
        
        if self.is_latent_quanted:
            LatentTable = torch.zeros_like(self.LatentTable)
            LatentTable.view(-1)[:] = self.latent_centroids[self.latent_labels]
        else:
            LatentTable = self.LatentTable
            
        n_v = coords.shape[1]
        LatentCode = LatentTable[indices,label,:] #(n_MLPs, batch_size, timeCodeDim)
        LatentCode = repeat(LatentCode,'n d -> n v d',v=n_v)
        output = coords
        output = torch.cat([output,LatentCode],dim=-1)
        for l_idx,mod in enumerate(self.net):
            output = mod(output, indices)
        return output
        
#*------------------------------------------------------*#
class DummyLinearWithChannel(nn.Module):
    '''
        Implements sinusoidal activations with multiple channel input
    '''
    def __init__(self, in_features, out_features, n_channels):
        super().__init__()
        
        self.in_features = in_features
        
        self.linear = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels)
        self.init_weights()
    
    def apply_weight_share(self,centroids,labels,_weight_mask=None):
        self.linear._weight_share(centroids,labels,_weight_mask=_weight_mask)
    
    def apply_local_weight_share(self,centroids,labels,_weight_mask=None):
        self.linear._local_weight_share(centroids,labels,_weight_mask=_weight_mask)
    
    def apply_bias_share(self,centroids,labels,_bias_mask=None):
        self.linear._bias_share(centroids,labels,_bias_mask=_bias_mask)
    
    def init_weights(self):
        #* comment below and uncomment the least lines to go back to normal init method
        with torch.no_grad():
            bound = np.sqrt(1.0*6 / self.in_features) / 30
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)
        
    def forward(self, input, indices):
        #input --> (n_MLPs, batch_size, in_features) indices --> (n_MLPs) the activate MLPs
        return self.linear(input, indices)

class AdaptiveMultiSineLayer(nn.Module):
    '''
        Implements sinusoidal activations with multiple channel input
    '''
    def __init__(self, in_features, out_features, n_channels, is_first=False, 
                 omega_0=30, const=1.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.const = const
        
        self.in_features = in_features
        
        self.linear = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels)
        self.init_weights()
    
    def apply_weight_share(self,centroids,labels,_weight_mask=None):
        self.linear._weight_share(centroids,labels,_weight_mask=_weight_mask)
    
    def apply_local_weight_share(self,centroids,labels,_weight_mask=None):
        self.linear._local_weight_share(centroids,labels,_weight_mask=_weight_mask)
    
    def apply_bias_share(self,centroids,labels,_bias_mask=None):
        self.linear._bias_share(centroids,labels,_bias_mask=_bias_mask)
    
    def init_weights(self):                
        with torch.no_grad():
            if self.is_first:
                bound = self.const/self.in_features
                self.linear.weight.uniform_(-bound, bound)      
                self.linear.bias.uniform_(-bound, bound)
            else:
                bound = np.sqrt(self.const*6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.uniform_(-bound, bound)
        
    def forward(self, input, indices):
        #input --> (n_MLPs, batch_size, in_features) indices --> (n_MLPs) the activate MLPs
        return torch.sin(self.omega_0 * self.linear(input, indices)) 

class AdaptiveLinearWithChannel(nn.Module):
    '''
        Implementation modified from https://github.com/pytorch/pytorch/issues/36591
        
        Evaluate only selective channels
    '''
    def __init__(self, input_size, output_size, channel_size):
        super(AdaptiveLinearWithChannel, self).__init__()
        
        #initialize weights
        self.weight = torch.nn.Parameter(torch.zeros(channel_size,
                                                     input_size,
                                                     output_size))
        self.bias = torch.nn.Parameter(torch.zeros(channel_size,
                                                   1,
                                                   output_size))
        self.is_global_share = True
        # weight share & prune init
        self.is_weight_share = False
        self.centroids = None
        self.labels = None
        self.is_pruned = False #* this var is only used in forward process
        self._weight_mask = None
        
        # bias share & prune init
        self.is_bias_share = False
        self.centroids_bias = None
        self.labels_bias = None
        self.is_bias_pruned = False #* this var is only used in forward process
        self._bias_mask = None
        
        #change weights to kaiming
        self.reset_parameters(self.weight, self.bias)
        
    def _local_weight_share(self,centroids,labels,_weight_mask=None):
        self.is_global_share = False
        self.is_weight_share = True
        device = self.weight.device
        self._weight_mask = _weight_mask
        self.centroids = centroids.to(device)
        self.labels = labels.to(device)
        self.label_max = torch.max(labels)
        self.label_min = torch.min(labels)
        if _weight_mask is not None:
            self.is_pruned = True
    
    def _weight_share(self,centroids,labels,_weight_mask=None):
        self.is_weight_share = True
        device = self.weight.device
        self._weight_mask = _weight_mask
        self.centroids = torch.nn.Parameter(centroids).to(device)
        self.labels = torch.nn.Parameter(labels,requires_grad=False).to(device)
        if _weight_mask is not None:
            self.is_pruned = True
    
    def _bias_share(self,centroids_bias,labels_bias,_bias_mask=None):
        self.is_bias_share = True
        device = self.weight.device
        self._bias_mask = _bias_mask
        self.centroids_bias = torch.nn.Parameter(centroids_bias).to(device)
        self.labels_bias = torch.nn.Parameter(labels_bias,requires_grad=False).to(device)
        if _bias_mask is not None:
            self.is_bias_pruned = True
        
    def reset_parameters(self, weights, bias):
        #* comment below and uncomment the least lines to go back to normal init method
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
    
    def forward(self, x, indices):
        #* weight checking
        if self.is_global_share:
            if self.is_weight_share:
                weight = torch.zeros_like(self.weight)
                if self.is_pruned:
                    weight.view(-1)[self._weight_mask.view(-1)[:] != 0] = self.centroids[self.labels]
                else:
                    weight.view(-1)[:] = self.centroids[self.labels]
            else:
                weight = self.weight
            #* bias checking 
            if self.is_bias_share:
                bias = torch.zeros_like(self.bias)
                if self.is_bias_pruned:
                    bias.view(-1)[self._bias_mask.view(-1)[:] != 0] = self.centroids_bias[self.labels_bias]
                else:
                    bias.view(-1)[:] = self.centroids_bias[self.labels_bias]
            else:
                bias = self.bias
        else:
            if self.is_weight_share:
                weight = torch.zeros_like(self.weight)
                nMLPs = weight.shape[0]
                for i in range(nMLPs):
                    weight.view(nMLPs,-1)[i,:] = self.centroids[i,self.labels[i,:]]
            bias = self.bias
        return torch.bmm(x, weight[indices, :, :]) + bias[indices, :, :]
        

    
#*--------------KiloNeRF stuff-------------------*#
    
class LinearWithChannel(nn.Module):
    '''
        Implementation from https://github.com/pytorch/pytorch/issues/36591
    '''
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()
        
        #initialize weights
        self.weight = torch.nn.Parameter(torch.zeros(channel_size,
                                                     input_size,
                                                     output_size))
        self.bias = torch.nn.Parameter(torch.zeros(channel_size,
                                                   1,
                                                   output_size))
        
        #change weights to kaiming
        self.reset_parameters(self.weight, self.bias)
        
    def reset_parameters(self, weights, bias):
        
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
    
    def forward(self, x):
        mul_output = torch.bmm(x, self.weight)
        return mul_output + self.bias

class MultiSineLayer(nn.Module):
    '''
        Implements sinusoidal activations with multiple channel input
    '''
    
    def __init__(self, in_features, out_features, n_channels, is_first=False, 
                 omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        
        
        self.linear = LinearWithChannel(in_features, out_features, n_channels)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class MultiReLULayer(nn.Module):
    '''
        Implements sinusoidal activations with multiple channel input
    '''
    
    def __init__(self, in_features, out_features, n_channels, is_final=False, 
                 omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        
        
        self.in_features = in_features
        if is_final:
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU()
        
        self.linear = LinearWithChannel(in_features, out_features, n_channels)
        
    def forward(self, input):
        return self.act(self.linear(input))
    

class KiloNeRF(nn.Module):
    """
    Fitting siggnal with lots of small MLPs, no prune and multi-scale.
    """
    def __init__(self,n_MLPs,in_features,out_features,init_features,n_layers,act='sine',omega_0=30):
        super(KiloNeRF, self).__init__()
        self.n_layers = n_layers
        net = []
        for i in range(n_layers):
            if i == 0:
                if act == 'sine':
                    net.append(MultiSineLayer(in_features,init_features,n_MLPs,is_first=True,omega_0=omega_0))
                else:
                    net.append(MultiReLULayer(in_features,init_features,n_MLPs))
                
            elif i < n_layers-1:
                if act == 'sine':
                    net.append(MultiSineLayer(init_features,init_features,n_MLPs,omega_0=omega_0))
                else:
                    net.append(MultiReLULayer(init_features,init_features,n_MLPs))
        final_layer = LinearWithChannel(init_features, out_features, n_MLPs)
        
        with torch.no_grad():
        #! Note: init the final layer is very important, and it is also very important not using a acitvation function in the final layer
            final_layer.weight.uniform_(-np.sqrt(6 / (init_features)) / 30.0, np.sqrt(6 / (init_features)) / 30.0)
        net.append(final_layer)
              
        self.net = nn.Sequential(*net)
    
    def forward(self,coords):
        #coords --> (n_MLPs, batch_size, in_features)
        return self.net(coords)
    
    
class AdaptiveKiloNeRF(nn.Module):
    """
    Fitting siggnal with lots of small MLPs, no prune and multi-scale.
    """
    def __init__(self,n_MLPs,in_features,out_features,init_features,n_layers,omega_0=30):
        super(AdaptiveKiloNeRF, self).__init__()
        self.n_layers = n_layers
        net = []
        for i in range(n_layers):
            if i == 0:
                net.append(AdaptiveMultiSineLayer(in_features,init_features,n_MLPs,is_first=True,omega_0=omega_0))
            if i < n_layers - 1:
                net.append(AdaptiveMultiSineLayer(init_features,init_features,n_MLPs,omega_0=omega_0))
            else: #final layer
                # final_layer = AdaptiveLinearWithChannel(init_features, out_features, n_MLPs)
                final_layer = DummyLinearWithChannel(init_features, out_features, n_MLPs)
                with torch.no_grad():
                #! Note: init the final layer is very important, and it is also very important not using a acitvation function in the final layer
                    final_layer.linear.weight.uniform_(-np.sqrt(6 / (init_features)) / 30.0, np.sqrt(6 / (init_features)) / 30.0)
                    final_layer.linear.bias.uniform_(-np.sqrt(6 / (init_features)) / 30.0, np.sqrt(6 / (init_features)) / 30.0)
                net.append(final_layer)
        self.net = nn.ModuleList(net)
    
    def forward(self,coords,indices):
        #coords --> (n_MLPs, batch_size, in_features)
        #indices --> (n_MLPs) Long Tensor
        output = coords[indices, ...]
        for mod in self.net:
            output = mod(output, indices)
        return output


class AdaptiveOmegaSineLayer(nn.Module):
    def __init__(self, in_features, out_features, n_channels, is_first=False, 
                 omega_0=3.0, const=1.0):
        super().__init__()
        # self.omega_0 = omega_0
        self.is_first = is_first
        self.const = const
        self.init_omega_0 = omega_0
        self.in_features = in_features
        
        self.linear = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels)
        self.omega_0 = nn.Parameter(torch.ones(n_channels,1,1)*omega_0)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = self.const/self.in_features
                self.linear.weight.uniform_(-bound, bound)      
                self.linear.bias.uniform_(-bound, bound)
            else:
                bound = np.sqrt(self.const*6 / self.in_features) / 30.0
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.uniform_(-bound, bound)
        
    def forward(self, input, indices, weight=None):
        #input --> (n_MLPs, batch_size, in_features) indices --> (n_MLPs) the activate MLPs
        return torch.sin(self.omega_0 * self.linear(input, indices, weight=weight)) 