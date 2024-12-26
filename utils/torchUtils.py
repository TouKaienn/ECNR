import os
import numpy as np
import yaml
from pathlib import Path
import shutil
import time
from tqdm import tqdm
import torch.nn.functional as F
import sys
from PIL import Image
import cv2
import subprocess
import json
import torch
from torch import nn
import torch.nn.utils.prune as prune 
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
import math
import re

import struct
from icecream import ic
from sklearn.cluster import KMeans

#*--------------------------------------------------------------------------------------------------*#
#* FileName: torchUtils.py
#* Last Modified: 2023-05-20
#* This is the torch utils libs to process the torch operation like model compression, etc.
#*--------------------------------------------------------------------------------------------------*#
def voxel_shuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width, in_depth = input.size()
    channels //= upscale_factor ** 3

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor
    out_depth = in_depth * upscale_factor

    input_view = input.reshape(batch_size, channels, upscale_factor, upscale_factor, upscale_factor, in_height, in_width, in_depth)

    return input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(batch_size, channels, out_height, out_width, out_depth)

class VoxelShuffle(nn.Module):
	def __init__(self,inchannels,outchannels,upscale_factor):
		super(VoxelShuffle,self).__init__()
		self.upscale_factor = upscale_factor
		self.conv = nn.Conv3d(inchannels,outchannels*(upscale_factor**3),3,1,1)

	def forward(self,x):
		x = voxel_shuffle(self.conv(x),self.upscale_factor)
		return x

def print_model_size(mdl):
    resend2CUDA = False
    if next(mdl.parameters()).device != 'cpu':
        mdl=mdl.cpu()
        resend2CUDA = True
    torch.save(mdl.state_dict(), "tmp.pt")
    model_size_MB = os.path.getsize("tmp.pt")/1e6
    # print("%.2f MB".format(model_size_MB))
    os.remove('tmp.pt')
    if resend2CUDA:
        mdl.cuda()
    return model_size_MB
    
class MultiSequential(nn.Sequential):
    '''
    nn.Sequential() does not support multiple inputs, this class is to solve this problem
        https://github.com/pytorch/pytorch/issues/19808#
    '''
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input

def seed_everything(seed=42):
    """This function will remove the initial uncertainty when init the neural network

    Args:
        seed (int): seed used to init the random components
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def kmeans_quantization(w,q):
    weight_feat = w.view(-1).unsqueeze(1).numpy()
    kmeans = KMeans(n_clusters=q,n_init=4).fit(weight_feat)
    return kmeans.labels_.tolist(),kmeans.cluster_centers_.reshape(q).tolist()

def get_weight_mats(net):
    weight_mats = [(name,parameters.data) for name, parameters in net.named_parameters() if re.match(r'.*.weight', name, re.I)]
    return [mat[1].cpu() for mat in weight_mats]

def get_bias_vecs(net):
    bias_vecs = [(name,parameters.data) for name, parameters in net.named_parameters() if re.match(r'.*.bias', name, re.I)]
    return [bias[1].cpu() for bias in bias_vecs]


def ints_to_bits_to_bytes(all_ints,n_bits):
    f_str = '#0'+str(n_bits+2)+'b'
    bit_string = ''.join([format(v, f_str)[2:] for v in all_ints])
    n_bytes = len(bit_string)//8
    the_leftover = len(bit_string)%8>0
    if the_leftover:
        n_bytes+=1
    the_bytes = bytearray()
    for b in range(n_bytes):
        bin_val = bit_string[8*b:] if b==(n_bytes-1) else bit_string[8*b:8*b+8]
        the_bytes.append(int(bin_val,2))
    #
    return the_bytes,the_leftover
#*-----------Model Compression-----------*#
class SirenEncoder:
    def __init__(self,net):
        self.net = net

    def encode(self,filename,n_bits,d_in=3):
        n_clusters = int(math.pow(2,n_bits))

        weight_mats = get_weight_mats(self.net)
        bias_vecs = get_bias_vecs(self.net)
        
        file = open(filename,'wb')


        # first layer: matrix and bias
        w_pos,b_pos = weight_mats[0].view(-1).tolist(),bias_vecs[0].view(-1).tolist()
        w_pos_format = ''.join(['f' for _ in range(len(w_pos))])
        b_pos_format = ''.join(['f' for _ in range(len(b_pos))])
        first_layer = file.write(struct.pack(w_pos_format, *w_pos))
        first_layer += file.write(struct.pack(b_pos_format, *b_pos))

        # middle layers: cluster, store clusters, then map matrix indices to indices
        mid_bias,mid_weight=0,0
        for weight_mat,bias_vec in zip(weight_mats[1:-1],bias_vecs[1:-1]):
            labels,centers = kmeans_quantization(weight_mat,n_clusters)

            # weights
            w = centers
            w_format = ''.join(['f' for _ in range(len(w))])
            mid_weight += file.write(struct.pack(w_format, *w))
            weight_bin,is_leftover = ints_to_bits_to_bytes(labels,n_bits)
            mid_weight += file.write(weight_bin)

            # encode non-pow-2 as 16-bit integer
            if n_bits%8 != 0:
                mid_weight += file.write(struct.pack('I', labels[-1]))
            #

            # bias
            b = bias_vec.view(-1).tolist()
            b_format = ''.join(['f' for _ in range(len(b))])
            mid_bias += file.write(struct.pack(b_format, *b))
        #

        # last layer: matrix and bias
        w_last,b_last = weight_mats[-1].view(-1).tolist(),bias_vecs[-1].view(-1).tolist()
        w_last_format = ''.join(['f' for _ in range(len(w_last))])
        b_last_format = ''.join(['f' for _ in range(len(b_last))])
        last_layer = file.write(struct.pack(w_last_format, *w_last))
        last_layer += file.write(struct.pack(b_last_format, *b_last))

        file.flush()
        file.close()
    



#*-----------Model Decompression-----------*#


class SirenDecoder:
    def __init__(self):
        pass
    #

    def decode(self,filename,model):
        

        file = open(filename,'rb')
        weight_mats_sample = get_weight_mats(model)
        bias_vecs_sample = get_bias_vecs(model)
        
        
        net = model

        # first layer: matrix and bias
        w_pos_format = ''.join(['f' for _ in range(self.d_in*self.layers[0])])
        b_pos_format = ''.join(['f' for _ in range(self.layers[0])])
        w_pos = torch.FloatTensor(struct.unpack(w_pos_format, file.read(4*self.d_in*self.layers[0])))
        b_pos = torch.FloatTensor(struct.unpack(b_pos_format, file.read(4*self.layers[0])))

        all_ws = [w_pos]
        all_bs = [b_pos]

        # middle layers: cluster, store clusters, then map matrix indices to indices
        total_n_layers = 2*(self.n_layers-1) if self.is_residual==1 else self.n_layers-1
        for ldx in range(total_n_layers):
            # weights
            n_weights = self.layers[0]*self.layers[0]
            weight_size = (n_weights*self.n_bits)//8
            if (n_weights*self.n_bits)%8 != 0:
                weight_size+=1
            c_format = ''.join(['f' for _ in range(self.n_clusters)])
            centers = torch.FloatTensor(struct.unpack(c_format, file.read(4*self.n_clusters)))
            inds = file.read(weight_size)
            bits = ''.join(format(byte, '0'+str(8)+'b') for byte in inds)
            w_inds = torch.LongTensor([int(bits[self.n_bits*i:self.n_bits*i+self.n_bits],2) for i in range(n_weights)])

            if self.n_bits%8 != 0:
                next_bytes = file.read(4)
                w_inds[-1] = struct.unpack('I', next_bytes)[0]
            #

            # bias
            b_format = ''.join(['f' for _ in range(self.layers[0])])
            bias = torch.FloatTensor(struct.unpack(b_format, file.read(4*self.layers[0])))

            w_quant = centers[w_inds]
            all_ws.append(w_quant)
            all_bs.append(bias)
        #

        # last layer: matrix and bias
        w_last_format = ''.join(['f' for _ in range(self.d_out*self.layers[-1])])
        b_last_format = ''.join(['f' for _ in range(self.d_out)])
        w_last = torch.FloatTensor(struct.unpack(w_last_format, file.read(4*self.d_out*self.layers[-1])))
        b_last = torch.FloatTensor(struct.unpack(b_last_format, file.read(4*self.layers[-1])))

        all_ws.append(w_last)
        all_bs.append(b_last)

        wdx,bdx=0,0
        for name, parameters in net.named_parameters():
            if re.match(r'.*.weight', name, re.I):
                w_shape = parameters.data.shape
                parameters.data = all_ws[wdx].view(w_shape)
                wdx+=1
            #
            if re.match(r'.*.bias', name, re.I):
                b_shape = parameters.data.shape
                parameters.data = all_bs[bdx].view(b_shape)
                bdx+=1
            #
        #

        return net
    


