import os
import numpy as np
import yaml
import torch
from pathlib import Path
import shutil
import time
from tqdm import tqdm
from logger import *
import torch.nn.functional as F
import sys
import json
from torchsummary import summary
from utils import *
from einops import rearrange, repeat, reduce
import torch.nn as nn
# from sklearn.cluster import KMeans
from fast_pytorch_kmeans import KMeans,MultiKMeans
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torchvision.transforms as transforms
import copy
from functools import wraps
import time



class EquallySizedKMeans:
    def __init__(self, n_clusters,scale,minibatch=2560000):
        self.n_clusters = n_clusters
        self.minibatch = minibatch
        self.scale_idx = int(np.log2(scale))
    
    def visualize(self, data):
        #* in the case you want to visualize the data, but the color may not be correct
        from sklearn.manifold import TSNE
        data_tmp = copy.deepcopy(data)
        data_2d = TSNE(n_components=2).fit_transform(data_tmp.cpu().numpy())
        data_2d = torch.tensor(data_2d,dtype=torch.float32).to(data.device)
        centroids, labels = self.fit(data_2d)
        print(labels.shape)
        color = np.random.rand(centroids.shape[0],3)
        factor = np.ceil(data.shape[0]/centroids.shape[0]).astype(np.int32)
        # data_color = repeat(color,'b c -> (b n) c',n=factor)
        data_2d = data_2d.cpu().numpy()
        # print(data_color)
        
        plt.figure()
        for i in range(centroids.shape[0]):
            plt.scatter(centroids[i, 0], centroids[i, 1],color=color[i],s=1000,alpha=0.5)
            plt.scatter(data_2d[labels==i, 0], data_2d[labels==i, 1],color=color[i])
        plt.show()
        exit()
    
    def fit(self, data, Tmax=1, n_cluster=None):
        n_cluster = self.n_clusters if n_cluster is None else n_cluster
        # Compute desired cluster size
        desired_size = len(data) // n_cluster
        soft_upper_limit_min = int(desired_size)
        soft_upper_limit_max = int(desired_size) + 1
        
        # Initialize means with k-means++
        minibatch = data.shape[0] if data.shape[0] < self.minibatch else self.minibatch
        init_kmeans = KMeans(n_clusters=n_cluster,mode='euclidean',minibatch=minibatch)
        # data_ls = torch.split(data,minibatch,dim=0)
        # print(data_ls[0].shape)
        # for j in data_ls:
        #     print(j.shape)
        # exit()
        # for i in range(np.ceil(data.shape[0]/minibatch).astype(np.int32)):
        init_kmeans.fit(data)
        labels = init_kmeans.predict(data)
        centroids = init_kmeans.centroids
        
        # Assign points based on the initialization method described
        data = data.cpu().numpy().astype(np.float32)
        labels = labels.cpu().numpy().astype(np.int32)
        centroids = centroids.cpu().numpy().astype(np.int32)
        
        # In case there is empty clusters
        clusterSize_cluter = np.array([np.sum(labels==ul).item() for ul in range(self.n_clusters)])
        # There is cases that some cluster is empty, we need to fill them with the largest cluster
        for i in range(n_cluster):
            if i in labels:
                centroids[i] = data[labels == i].mean(axis=0)
                clusterSize_cluter[i] /= 2
            else:
                clusterSize_descending_idx = np.argsort(clusterSize_cluter)
                centroids[i] = centroids[clusterSize_descending_idx[0]]
                clusterSize_cluter[clusterSize_descending_idx[0]] /= 2 # half the largest cluster size
        
        for _ in range(Tmax):  # Max 10 iterations, can be changed
            print("clustering progress: ",_)
            # for i in range(n_cluster):
            #     centroids[i] = data[labels == i].mean(axis=0)
            #in case run out of memory
            centroids_square = (centroids**2).sum(axis=1).astype(np.float32)
            data_square = (data**2).sum(axis=1).astype(np.float32)
            dot_product = np.dot(centroids, data.T).astype(np.float32)
            distances = data_square + centroids_square[:, np.newaxis] - 2*dot_product
            
            # Compute the delta for sorting
            min_distances = distances.min(axis=0)
            # print(min_distances.shape) # 1000
            
            # second_min_distances = np.partition(distances, 1, axis=0)[1]
            second_min_distances = distances.max(axis=0)
            deltas = second_min_distances - min_distances

            # Sort elements by delta
            sorted_idx = np.argsort(deltas)[::-1] #if delta is large, then they should be sort first
            
            # Reassign labels
            labels = np.empty_like(labels)
            # print(labels)
            cluster_sizes = np.zeros(n_cluster, dtype=int)
            for idx in sorted_idx: # 1000
                best_clusters = np.argsort(((data[idx] - centroids)**2).sum(axis=1)) # distance from close to far for point idx
                all_full = True
                for cluster in best_clusters: # 10
                    if cluster_sizes[cluster] < soft_upper_limit_min:
                        all_full = False
                        labels[idx] = cluster
                        cluster_sizes[cluster] += 1
                        break
                if all_full:
                    for cluster in best_clusters:
                        if cluster_sizes[cluster] < soft_upper_limit_max:
                            labels[idx] = cluster
                            cluster_sizes[cluster] += 1
                            break
            
            for i in range(n_cluster):
                centroids[i] = data[labels == i].mean(axis=0)
        
        return centroids, labels

def calNumBitsNeedsForIdx(data):
    #* calculate the number of bits needed to represent the data (assume it is larger than 0)
    data = data.flatten()
    max_val = np.max(data)
    return np.ceil(np.log2(max_val+1)).astype(np.int32)
      
    

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time_ns()/1e9
        result = f(*args, **kw)
        te = time.time_ns()/1e9
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

def needSpecInterpolateEndFrame(total_samples,scale):
    """check if the end frames need to be interpolated with 2 frames instead of 1 frame"""
    while(scale != 1):
        total_samples = len([i for i in range(1,total_samples+1,2)])
        scale = scale//2
    print(total_samples)
    if total_samples%2 == 0:
        return True
    else:
        return False

if __name__ == "__main__":
    a = needSpecInterpolateEndFrame(15,2)
    print(a)
    
def linear_interpolation(dir_path,save_dir_path,total_samples,scale):
    """doing linear interpolation for the dir_path, then save the result to save_dir_path

    Args:
        total_samples (int): total samples of the datasets
        dir_path (str): the result path used to save previous results
    """
    # print("total_samples: ",total_samples
        #   ,"scale: ",scale)
    # print("needSpecInterpolateEndFrame: ",needSpecInterpolateEndFrame(total_samples,scale))
    vol_paths = getFilePathsInDir(dir_path)
    baseName = os.path.basename(vol_paths[0])[:-8]    
    vols = []
    idx = 1
    # read all vols data from prev results
    for p in vol_paths:
        vols.append(readDat(p))
    # interpolated_vols = [vols[0]]
    saveDat(vols[idx-1],os.path.join(save_dir_path,baseName+f"{idx:04d}.raw"))
    # print(f"save time {idx}")
    # print(len(vols))
    # print("spec interpolate: ",needSpecInterpolateEndFrame(total_samples,scale))
    for i in range(1,len(vols),1):
        v_s = vols[i-1]
        v_e = vols[i]
        
        if (i == (len(vols) - 1)) and needSpecInterpolateEndFrame(total_samples,scale):
            v_i_1 = (2/3)*v_s+(1/3)*v_e
            v_i_2 = (1/3)*v_s+(2/3)*v_e
            idx += 1
            saveDat(v_i_1,os.path.join(save_dir_path,baseName+f"{idx:04d}.raw"))
            # print(f"save interpolated 1 time {idx}",i)
            idx += 1
            saveDat(v_i_2,os.path.join(save_dir_path,baseName+f"{idx:04d}.raw"))
            # print(f"save interpolated 2 time {idx}",i)
        else:
            v_i = (v_s+v_e)/2
            idx += 1
            saveDat(v_i,os.path.join(save_dir_path,baseName+f"{idx:04d}.raw"))
            # print(f"save interpolated time {idx}",i)
        idx += 1
        saveDat(v_e,os.path.join(save_dir_path,baseName+f"{idx:04d}.raw"))
        # print(f"save end time {idx}")


def gaussian(data,window_size=5):
    blur_data = None
    data = torch.Tensor(data).unsqueeze(0)
    data = transforms.GaussianBlur(window_size)(data)
    blur_data = data.squeeze().numpy()
    return blur_data

def resizeV(vol,scale_factor,flatten=True,keep_range=True,mode='trilinear'):
    """only support numpy, resize vol, vol should be flatten 1D array without transpose

    Args:
        vol (_type_): _description_
        source_shape (_type_): _description_
        target_shape (_type_): _description_
        flatten (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    v_max,v_min = np.max(vol),np.min(vol)
    v_3d_resize = torch.nn.functional.interpolate(torch.from_numpy(vol[None,None,...]),scale_factor=scale_factor,mode=mode).squeeze().numpy()
    if flatten:
        v_resize = v_3d_resize.flatten('F')
    else:
        v_resize = v_3d_resize
    if keep_range:
        v_resize = normalizeVol(vol=v_resize,outMin=v_min,outMax=v_max)
    return v_resize

def sampleV(vol,source_shape,scale_factor,flatten=True,keep_range=True):
    """only support numpy,sample vol, vol should be flatten 1D array without transpose

    Args:
        vol (_type_): _description_
        source_shape (_type_): _description_
        target_shape (_type_): _description_
        flatten (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    v_max,v_min = np.max(vol),np.min(vol)
    # print(vol.shape)
    if isinstance(scale_factor,int):
        scale_x,scale_y,scale_z = scale_factor,scale_factor,scale_factor
    else:
        scale_x, scale_y, scale_z = scale_factor
    target_shape = (source_shape[0]//scale_x,source_shape[1]//scale_y,source_shape[2]//scale_z)
    if len(vol.shape) <= 2:
        vol = vol.reshape(source_shape).transpose()
    v = []
    for x in range(0,source_shape[0], scale_x):
        for y in range(0,source_shape[1], scale_y):
            for z in range(0,source_shape[2], scale_z):
                v.append(vol[x,y,z])
    v = np.array(v).reshape(target_shape)
    if flatten:
        v_sample = v
    else:
        v_sample = v.reshape(target_shape)
    if keep_range:
        v_sample = normalizeVol(vol=v_sample,outMin=v_min,outMax=v_max)
    return v_sample

def multikmeans_quantization(w,q):
    w_shape = w.shape
    # weight_feat = w.view(-1).unsqueeze(1)
    weight_feat = w.view(w.shape[0],-1).unsqueeze(-1)
  
    kmeans = MultiKMeans(n_clusters=q,mode='euclidean')# operate in parallel
    labels = kmeans.fit_predict(weight_feat).reshape(w_shape).to(w.device)
    centroids = kmeans.centroids.reshape((w_shape[0],q))
    
    return labels,centroids

def kmeans_quantization(w,q):
    w_shape = w.shape
    # print(w_shape)
    weight_feat = w.view(-1).unsqueeze(1)
    kmeans = KMeans(n_clusters=q,mode='euclidean')#.fit(weight_feat)
    labels = kmeans.fit_predict(weight_feat).reshape(w_shape).to(w.device)
    centroids = kmeans.centroids.reshape(q)
    
    # exit()
    return labels,centroids

def ints_to_bits_to_bytes(ints,n_bits):
    """First, convert list with int number to bits, then convert bits to bytes by complementing the bits to 8*n bits.
    Args:
        ints (list[int]): list contain int number
        n_bits (int): n_bits of each int number
    """
    f_str = '#0'+str(n_bits+2)+'b' # we use #0xb to avoid the case where first bit is 0, then the format will ignore it
    bit_string = ''.join(format(b, f_str)[2:] for b in ints)
    n_bytes = len(bit_string)//8
    the_leftover = len(bit_string)%8 > 0
    if the_leftover:
        n_bytes += 1
    the_bytes = bytearray() #* bytearray can only accept byte with value from 0 to 255, so we have to encode in 8-bit format
    for b in range(n_bytes):
        bin_val = bit_string[8*b:] if b==(n_bytes-1) else bit_string[8*b:8*b+8] # byte by byte, then the last one will be the leftover
        the_bytes.append(int(bin_val,2))
    return the_bytes, the_leftover

def one_bit_decode(byte,length):
    """decode one bit from byte
    """
    left_over = length%8
    bits = ''.join(format(b, f'08b') for b in byte)
    if left_over:
        ls = [int(bits[i]) for i in range(len(bits)-8)]
        left_over_bits = bits[-left_over:]
        ls += [int(left_over_bits[i]) for i in range(len(left_over_bits))]
    else:
        ls = [int(bits[i]) for i in range(len(bits))]
    return ls

def bytes_to_bits_to_int(byte,n_bits):
    is_leftover = False
    n_elements = (len(byte)*8)//n_bits
    n_byte = len(byte)//8
    if (len(byte)*8)%n_bits > 0:
        is_leftover = True
        n_elements -= 1
    bits = ''.join(format(b, f'08b') for b in byte)
    ls = [int(bits[n_bits*i:n_bits*(i+1)],2) for i in range(n_elements)]
    if is_leftover:
        left_over_bits = (bits[n_bits*n_elements:-8]+bits[-n_bits+len(bits[n_bits*n_elements:-8]):])[:n_bits]
        ls.append(int(left_over_bits,2))
    return ls

def parseDatasetInfo(dataset_varList,dataHeaderJsonPath):
    #*dataset should have a form like DatasetName_VarName
    def getVarDatasetInfo(DatasetName,VarName,dataHeaderJsonPath):
        dataHeader = json_loader(dataHeaderJsonPath)
        varInfo = dataHeader[DatasetName]
        del varInfo['vars']
        varInfo['data_path'] = varInfo['data_path'][VarName]
        varInfo['total_time_steps'] = [i for i in range(1,varInfo['total_samples']+1)] if varInfo['total_samples'] > 1 else [1]
        return varInfo
    d = {dataset_var:{} for dataset_var in dataset_varList}
    for i in dataset_varList:
        datasetName,varName = i.split("_")
        d[i]=getVarDatasetInfo(datasetName,varName,dataHeaderJsonPath)
    return d

def adjust_lr(optimizer,new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
            
def setup_Exp(opt=None):
    """setup the settings and logger, create Exp files
    """
    #parse the settings
    args = yaml_loader(opt.config_path)
    mode = args['mode']
    model_setting = args['model_setting']
    train_setting = args['train_setting']
    data_setting = args['data_setting']
    version =f"v{train_setting['version']}"
    nums_sacle = data_setting['nums_scale']

    # add some useful attribute to args
    log_base_dirName = version# getYMD() + version
    args['log_base_dir'] = os.path.join(train_setting['log_root_dir'],log_base_dirName) if opt.out is None else opt.out

    #del and then create log base dir, result dir and their sub-dirs
    ensure_dirs(args['log_base_dir'])
    for sub_dirs in ['Log','Results']:
        ensure_dirs(os.path.join(args['log_base_dir'],sub_dirs))
        delDirsInDir(os.path.join(args['log_base_dir'],sub_dirs))
        if (sub_dirs == "Results" or sub_dirs == "ResultsRaw"):
            for var in data_setting['dataset']:
                for i in range(nums_sacle,-1,-1):
                    scale = 2**i
                    ensure_dirs(os.path.join(args['log_base_dir'],sub_dirs,var+f"-scale{scale}"))
    
    #move the model,setting and main file into base_dir
    copy_modelSetting(args['log_base_dir']) #move everything to log
    _, setting_file_name = os.path.split(opt.config_path)
    setting_savedTo = os.path.join(args['log_base_dir'],setting_file_name)
    if os.path.exists(setting_savedTo):
        os.remove(setting_savedTo)
    shutil.copy(opt.config_path,setting_savedTo)

    #setup logger
    logger = setup_logger(log_file_dir=os.path.join(args['log_base_dir'],'Log'))

    return args,train_setting,model_setting,data_setting,logger

def get_mgrid(sidelen, dim=2, s=1,t=0):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[1]:s, :sidelen[0]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[1] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[0] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[2]:s, :sidelen[1]:s, :sidelen[0]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[2] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[0] - 1)
    elif dim == 4:
        pixel_coords = np.stack(np.mgrid[:sidelen[0]:(t+1), :sidelen[3]:s, :sidelen[2]:s, :sidelen[1]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[3] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        pixel_coords[..., 3] = pixel_coords[..., 3] / (sidelen[1] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)
    pixel_coords -= 0.5
    pixel_coords *= 2.
    
    pixel_coords = np.reshape(pixel_coords,(-1,dim))
    return pixel_coords

def flattenDict(d):
    """flatten the nesting dict to a flatten dict
    """
    flatten_d = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flatten_d.update(flattenDict(value))
        else:
            flatten_d[key] = value
    return flatten_d


class UnFold(nn.Module):
    '''
        Unfolding operator for 3D tensors/array with disjoint folding. The operator
        is a drop in replacement for unfoldNd.UnfoldNd
        Because this function do not account for overlapping folding, this function can run faster than torch
        
        Inputs:
            block_size: Folding block size (xc,yc,zc)
            stride: Folding stride. This is just for compatibility, as the 
                stride is fixed to be the same as kernel_size
            tensor: (T,X,Y,Z) sized tensor (need reshape & transpose beforehand)
                
        Outputs:
            unfolded_tensor: (T, mul(block size), xc, yc, zc) sized unfolded tensor
    '''
    def __init__(self,block_size,stride=None):
        super(UnFold, self).__init__()
        self.bsize = block_size
        self.stride = stride
     
    
    def forward(self,tensors,flatten_out=False):
        if(len(tensors.shape) == 3):
            unfolded_tensors = rearrange(tensors, '(nx xc) (ny yc) (nz zc) -> (nx ny nz) xc yc zc',
                                     xc=self.bsize[0], yc=self.bsize[1], zc=self.bsize[2])
            if flatten_out:
                N = unfolded_tensors.shape[0]
                unfolded_tensors = unfolded_tensors.transpose((0,3,2,1)).reshape(N,-1)
        elif(len(tensors.shape) == 4):
            unfolded_tensors = rearrange(tensors, 'T (nx xc) (ny yc) (nz zc) -> T (nx ny nz) xc yc zc',
                                     xc=self.bsize[0], yc=self.bsize[1], zc=self.bsize[2])
            if flatten_out:
                T, N = unfolded_tensors.shape[0], unfolded_tensors.shape[1]
                unfolded_tensors = unfolded_tensors.transpose((0,1,4,3,2)).reshape(T,N,-1)
        else:
            raise ValueError(f'UnFold: Input tensor must be 3D or 4D, but got dims: {tensors.shape}')
        return unfolded_tensors
        
class Fold(nn.Module):
    '''
        Folding operator for 3D tensors with disjoint folding. The operator
        is a drop in replacement for unfoldNd.FoldNd
        Because this function do not account for overlapping folding, this function can run faster than torch
        
        Inputs:
            block_size: Folding block size
            stride: Folding stride. This is just for compatibility, as the 
                stride is fixed to be the same as kernel_size
            unfolded_tensor: (T, mul(block size), xc, yc, zc) sized unfolded tensor
                
        Outputs:
            tensor: (T,X,Y,Z) sized tensor (need flatten('F') & tofile(format = '<f') afterwards)
    '''
    def __init__(self,output_size,block_size,stride=None):
        super(Fold, self).__init__()
        self.bsize = block_size
        self.output_size = output_size
        self.stride = stride
    
    def forward(self,unfolded_tensors,flatten_in=False,flatten_out=False):
        folded_tensors = None
        if((len(unfolded_tensors.shape) == 3 and not flatten_in) or (len(unfolded_tensors.shape) == 2 and flatten_in)):
            if flatten_in:
                unfolded_tensors = rearrange(unfolded_tensors, 'n (xc yc zc) -> n xc yc zc',
                                             xc=self.bsize[0], 
                                             yc=self.bsize[1], 
                                             zc=self.bsize[2])
                unfolded_tensors = rearrange(unfolded_tensors, 'n xc yc zc -> n zc yc xc')
            folded_tensors = rearrange(unfolded_tensors, '(nx ny nz) xc yc zc -> (nx xc) (ny yc) (nz zc)',
                                        xc=self.bsize[0], 
                                        yc=self.bsize[1], 
                                        zc=self.bsize[2],
                                        nx=self.output_size[0]//self.bsize[0],
                                        ny=self.output_size[1]//self.bsize[1],
                                        nz=self.output_size[2]//self.bsize[2])
            if flatten_out:
                if type(folded_tensors) == torch.Tensor:
                    folded_tensors = folded_tensors.transpose(2,0).reshape(-1)
                elif type(folded_tensors) == np.ndarray:
                    folded_tensors = folded_tensors.transpose((2,1,0)).reshape(-1)
            
        elif((len(unfolded_tensors.shape) == 4 and not flatten_in) or (len(unfolded_tensors.shape) == 3 and flatten_in)):
            if flatten_in:
                unfolded_tensors = rearrange(unfolded_tensors, 'T n (xc yc zc) -> T n xc yc zc',
                                             xc=self.bsize[0], 
                                             yc=self.bsize[1], 
                                             zc=self.bsize[2])
                unfolded_tensors = rearrange(unfolded_tensors, 'T n xc yc zc -> T n zc yc xc')
            folded_tensors = rearrange(unfolded_tensors, 'T (nx ny nz) xc yc zc -> T (nx xc) (ny yc) (nz zc)',
                                        xc=self.bsize[0], 
                                        yc=self.bsize[1], 
                                        zc=self.bsize[2],
                                        nx=self.output_size[0]//self.bsize[0],
                                        ny=self.output_size[1]//self.bsize[1],
                                        nz=self.output_size[2]//self.bsize[2])
            if flatten_out:
                T = folded_tensors.shape[0]
                folded_tensors = folded_tensors.transpose((0,3,2,1)).reshape(T,-1)
        return folded_tensors