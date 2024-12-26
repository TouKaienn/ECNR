import numpy as np
import torch
import skimage 
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import data,img_as_float
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import random
from utils import *
import re
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import data,img_as_float
from collections import OrderedDict
from tools import *
from einops import rearrange, repeat, reduce, pack, unpack
import copy
from fast_pytorch_kmeans import KMeans

#-------The lib below is not functional----------#
from pprint import pprint
#These three lines are just for debugging
import matplotlib.pyplot as plt
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)
np.set_printoptions(threshold=sys.maxsize)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.seterr(divide='ignore', invalid='ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiScaleScalarDataSet():
    def __init__(self,args,scale,idx,logger=None):
        self.data_setting = args['data_setting']
        self.mode = args['mode']
        self.block_dims = args['model_setting']['block_dims'][idx]
        self.batch_size = self.data_setting['batch_size']#[idx]
        self.logger = logger.getLogger("trainLogger")
        # self.batch_size = (self.block_dims[0]*self.block_dims[1]*self.block_dims[2]//30)
        self.bottom_scale = 2**self.data_setting['nums_scale']
        self.cut_threshold = self.data_setting['cut_threshold']
        datasetInfoPath = "./dataInfo/localDataInfo.json" if self.mode in ['debug', 'local_inf'] else './dataInfo/CRCDataInfo.json'
        self.datasetInfoDict = json_loader(datasetInfoPath)
        self.datasetInfo = parseDatasetInfo(self.data_setting['dataset'],datasetInfoPath)
        self.dataset_varList = self.data_setting['dataset']
        self.total_samples = self.datasetInfo[self.dataset_varList[0]]['total_samples']
        self.log_base_dir = args['log_base_dir']
        self.Results_path = os.path.join(self.log_base_dir,'Results')
        self.numBin = 1 # init numBin, will be updated in ReadData and then used to init the model
        # self.n_blocks_eachMLP = args['train_setting']['n_blocks_eachMLP'][idx]
        self.scale = scale
        for d_v in self.dataset_varList:#cur downscaled dims
            self.datasetInfo[d_v]['cur_dim'] = [d//self.scale for d in self.datasetInfo[d_v]['dim']] if self.scale!=1 else self.datasetInfo[d_v]['dim']
            self.datasetInfo[d_v]['next_dim'] = [d*2 for d in self.datasetInfo[d_v]['cur_dim']] if self.scale!=1 else self.datasetInfo[d_v]['dim']
        
        self.downScaleMethod = self.data_setting['downScaleMethod']
        self.data_Min_Max = None
        
        self.inf_block_dims = [d*2 for d in self.block_dims] if scale != 1 else self.block_dims
        self.stride = [int(d) for d in self.block_dims] #* parameter size of MLP might became 2 times bigger than the original size when stride have overlap part
        self.inf_stride = [stride*2 for stride in self.stride] if scale != 1 else self.stride
        self.padding = getVolPaddingSize(self.datasetInfo[d_v]['cur_dim'], kernel_size=self.block_dims, stride=self.stride)
        self.inf_padding = getVolPaddingSize(self.datasetInfo[d_v]['next_dim'], kernel_size=self.inf_block_dims, stride=self.inf_stride)
        # inf_output_shape = [d + 2 * p for d, p in zip(self.datasetInfo[d_v]['next_dim'], self.inf_padding)]
        # self.Fold = {d_v:Fold3D(output_shape=inf_output_shape,kernel_size=self.inf_block_dims,stride=self.inf_stride,remove_padding=self.inf_padding) for d_v in self.dataset_varList}
        inf_output_shape = [d + 2 * p for d, p in zip(self.datasetInfo[d_v]['cur_dim'], self.padding)]
        
       
        #* Fold for same dimension inference
        self.Fold = {d_v:Fold3D(output_shape=inf_output_shape,kernel_size=self.block_dims,stride=self.stride,remove_padding=self.padding) for d_v in self.dataset_varList}
        self.UnFold = {d_v:UnFold3D(kernel_size=self.block_dims,stride=self.stride,padding=self.padding) for d_v in self.dataset_varList}

    def get_sampled_timeSteps(self,timeSteps,scale):
        if scale == 1:
            return timeSteps
        final_element = timeSteps[-1]
        # print("time steps: ",timeSteps)
        while(scale != 1):
            self.read_timeSteps = []
            for idx in range(0,len(timeSteps),2):
                self.read_timeSteps.append(timeSteps[idx])
                if(len(timeSteps) - 4 == idx):
                    self.read_timeSteps.append(timeSteps[len(timeSteps)-1])
                    break
            timeSteps = self.read_timeSteps
            scale //= 2
            # print(f"self.read_timeSteps at scale {scale}: ",self.read_timeSteps)
        if(self.read_timeSteps[-1] != final_element):
            self.read_timeSteps.append(final_element)
        return self.read_timeSteps   

    def getBlockNums(self):
     #*block dims can be different for different variables, but block nums should be the same for all variables
        n_blocks = None
        for d_v in self.dataset_varList:
            d,v = self.dataset_varList[0].split("_")
            cur_dim = self.datasetInfo[d_v]['cur_dim']
            test_sample = torch.ones(1,1,cur_dim[0],cur_dim[1],cur_dim[2])
            n_blocks = self.UnFold[d_v](test_sample).shape[0]
        return n_blocks

    def keepTopKBlocks(self,block_loss,k): #data (t,n_MLPs,v)
        # this functin keeps the topk blocks data for each time step
        n_MLPs = self.data[self.dataset_varList[0]].shape[1]
        learn_data = []
        for d_v in self.dataset_varList:
            d,v = d_v.split("_")
            for idx,t in enumerate(range(self.num_timeStep)):
                values,learn_indices_shuffle = torch.topk(block_loss[idx],k[idx])
                learn_indices, _ = torch.sort(learn_indices_shuffle)
                master_mask = torch.ones(n_MLPs,dtype=torch.bool)
                master_mask[learn_indices] = False
                self.optimize_blocks[idx,master_mask] = 0
                learn_data += list(self.data[d_v][idx,learn_indices,:])
            self.data[d_v] = np.array(learn_data)
    
    def _dataNormalize(self):
        #must call after keepTopKBlocks
        # print(self.data[self.dataset_varList[0]].shape)
        n_total_blocks,n_v = self.data[self.dataset_varList[0]].shape
        
        self.data_Min_Max = {dataset_var:np.zeros((n_total_blocks,2)) for dataset_var in self.dataset_varList} # (t,n_MLPs,0) --> Min, (t,n_MLPs,1) --> Max
        for d_v in self.dataset_varList: #each block have one max and one min so it should be 64
            self.data_Min_Max[d_v][:,0] = self.data[d_v].min(axis=-1)
            self.data_Min_Max[d_v][:,1] = self.data[d_v].max(axis=-1)
            self.data[d_v] = 2.0*((self.data[d_v] - self.data_Min_Max[d_v][:,0,None])/(self.data_Min_Max[d_v][:,1,None] - self.data_Min_Max[d_v][:,0,None])-0.5)
            self.data[d_v][np.where(np.isnan(self.data[d_v]))] = 0.0
    
    def ReadData(self,scale,k,prev_resultRootDir=None): #prev_resultRootDir will be useless if you choose 'sample' mode
        self.data = {dataset_var:[] for dataset_var in self.dataset_varList}
        self.coords_each_MLP = {dataset_var:[] for dataset_var in self.dataset_varList} #This for train
        self.grid = {dataset_var:[] for dataset_var in self.dataset_varList} #This for inf
        self.target_v = {dataset_var:[] for dataset_var in self.dataset_varList} #*Debugging var
        self.prev_resultRootDir = {dataset_var:[] for dataset_var in self.dataset_varList} # for 'resize' mode
        self.sample_recon_resultRootDir = {dataset_var:[] for dataset_var in self.dataset_varList} # for 'sample' mode
        self.cur_resultRootDir = {dataset_var:[] for dataset_var in self.dataset_varList} # for 'sample' mode


        for d_v in self.dataset_varList:
            d,v = d_v.split("_")
            time_steps = self.datasetInfo[d_v]['total_time_steps']
            self.read_timeSteps = self.get_sampled_timeSteps(time_steps,scale) # sample the time steps from whole time steps
            self.num_timeStep = len(self.read_timeSteps)
            orig_dim = self.datasetInfo[d_v]['dim']
            cur_dim = self.datasetInfo[d_v]['cur_dim']
            data_path = self.datasetInfo[d_v]['data_path']
            self.sample_recon_resultRootDir[d_v] = os.path.join(self.Results_path,d_v)+f"-scale{1}"
            self.cur_resultRootDir[d_v] = os.path.join(self.Results_path,d_v)+f"-tmp-scale{scale}"

            self.read_timesteps_len = len(self.read_timeSteps)
            #* reinit data dict #(t,N,v) t-> timeBinSpan, N-> MLP nums, v-> block size
            #* finaly should be ((N), (t,v))
            # blockSize = self.block_dims[0]*self.block_dims[1]*self.block_dims[2] #*self.timeBinSpan
            self.total_blocks_nums = self.getBlockNums()*self.numBin #init MLP nums with block nums for cur scale
            self.optimize_blocks = torch.ones((self.read_timesteps_len,self.total_blocks_nums)) #size (timeSteps, n_MLPs): 1 means optimize, 0 means not optimize

            if prev_resultRootDir is not None:
                prev_resultDir = os.path.join(prev_resultRootDir,d_v)+f"-scale{scale*2}"
                prev_resPaths = getFilePathsInDir(dir_path=prev_resultDir,ext='raw')

            #* read data
            for idx,t in enumerate(self.read_timeSteps): #process data at each time step
                v = readDat(data_path+f'{t:04d}.raw')
                v = v.reshape(orig_dim[2],orig_dim[1],orig_dim[0]).transpose()
                scale_factor = scale
                source_shape = orig_dim
                while(scale_factor != 1):#downscale: high -> low
                    v = gaussian(v)
                    v = sampleV(v,source_shape=source_shape,scale_factor=2, flatten=False, keep_range=False)          
                    source_shape = [d//2 for d in source_shape]
                    scale_factor = scale_factor//2
                self.target_v[d_v].append(v)
                if prev_resultRootDir is not None:
                    prev_dim = [d//2 for d in self.datasetInfo[d_v]['cur_dim']]
                    pre_v_t = readDat(prev_resPaths[idx]).reshape(prev_dim[2],prev_dim[1],prev_dim[0]).transpose()
                    pre_v_t = resizeV(pre_v_t,scale_factor=2,flatten=False,keep_range=False)
                    v = v - pre_v_t
                # #*=====Debugging code=====*#
                # tmp_v = v.flatten('F')
                # tmp_v.tofile(f'res{idx:04d}.raw',format='<f')
                # print("res.raw updated")
                # #*========================*#
                if cur_dim != self.block_dims:    
                    v_unfold = self.UnFold[d_v](v,flatten_out=True)
                else:
                    v_unfold = v.flatten('F').reshape(1,-1)
                #*========================*#
                self.data[d_v].append(v_unfold) # gt data without prune
            
            self.data[d_v] = np.array(self.data[d_v])
            

            if scale != self.bottom_scale: #* prune MLPs at cur scale, prune along time dim
                v_residual_MSE_t = [] # Block-wise MSE loss at each time step
                self.n_MLPs_t = np.zeros(len(self.read_timeSteps),dtype=np.int32) # save how many MLPs we need for each time step
                for idx,t in enumerate(self.read_timeSteps):
                    v_residual_t = self.data[d_v][idx,:,:] # (n_MLPs, v)
                    MSE_t = np.mean(v_residual_t**2,axis=-1,keepdims=False) # block-wise MSE loss at cur time step
                    v_residual_MSE_t.append(torch.from_numpy(MSE_t))
                    learn_indices = torch.tensor(np.argwhere(MSE_t > self.cut_threshold)).squeeze() #indices which no need to learn
                    # print(f"time steps:{t} ",learn_indices)
                    self.n_MLPs_t[idx] = learn_indices.shape[0]
                self.keepTopKBlocks(v_residual_MSE_t,k=self.n_MLPs_t)
                self._dataNormalize()#*: MIN_MAX normalization to [-1,1] for the target dataset each blcoks
                # self.data[d_v] = rearrange(self.data[d_v], 't n v -> (t n) v') # this should be done in self.keeptopKBlocks
            #* let's comment the three lines below, assign n_MLPs_t in the next step
            else:
                self.n_MLPs_t = np.ones(self.num_timeStep,dtype=np.int32)*self.total_blocks_nums
                self.data[d_v] = rearrange(self.data[d_v], 't n v -> (t n) v')
                self._dataNormalize()#*: MIN_MAX normalization to [-1,1] for the target dataset each blcoks
                
            # print(self.n_MLPs_t)
            #input data should be shape (t N) v
            
            self.data[d_v],self.MLP_idx_each_t,self.latent_idx_each_t,self.unique_labels=self.blockClustering(k,data=self.data[d_v])
            
            # print("num MLPs for each time step: ",self.n_MLPs_t)
            
            self.data[d_v] = np.stack(self.data[d_v],axis=0) #(t,N,v)
            self.data[d_v] = rearrange(self.data[d_v], 't N v -> N (t v)')

            # print(self.block_dims)
            #*todo: Heck here, the name should not be self.coords_each_MLP, should be self.coords_perMLP
            self.coords_each_MLP[d_v] = torch.FloatTensor(get_mgrid([self.block_dims[0],self.block_dims[1],self.block_dims[2]],dim=3))
            bs,v = self.coords_each_MLP[d_v].shape #bs: block size, v: coord dim 3
            self.latentIdx_each_MLP = torch.arange(0,self.n_blocks_perMLP).unsqueeze(-1)
            self.coords_each_MLP[d_v] = repeat(self.coords_each_MLP[d_v],'b v -> (t b) v',t=self.n_blocks_perMLP)
            self.latentIdx_each_MLP = repeat(self.latentIdx_each_MLP,'t 1 -> (t b) 1',b=bs)
            # print("coords_perBlock shape: ",self.coords_each_MLP[d_v].shape)
            # print("timeCol_perBlock shape: ",self.latentIdx_each_MLP.shape)
            
            self.coords_each_MLP[d_v] = torch.cat([self.latentIdx_each_MLP,self.coords_each_MLP[d_v]],dim=-1)
            # print(self.coords_each_MLP[d_v])
            # print(self.coords_each_MLP[d_v].shape)
            self.grid[d_v] = get_mgrid([self.block_dims[0],self.block_dims[1],self.block_dims[2]],dim=3)
            # exit()
    def blockClustering(self,k,data):
        def _count_occurrences_2d(lst):
            counts = {}
            result = []
            for row in lst:
                row_res = []
                for item in row:
                    if item not in counts:
                        counts[item] = 0
                    else:
                        counts[item] += 1
                    row_res.append(counts[item])
                result.append(row_res)
            return result
        
        n_v = data.shape[-1]
        data = torch.tensor(data,dtype=torch.float32).to(device)
        
        if k != 1: # if k == 1, no need to do clustering
            # tic = time.time()
            n_clusters = int(np.ceil(data.shape[0]/k).astype(np.int32))
            # n_clusters = np.ceil(data.shape[0]/self.n_blocks_eachMLP).astype(np.int32)    
            kmeans = EquallySizedKMeans(n_clusters,scale=self.scale)
            # kmeans.visualize(data) #todo: if you want to visualize
            # print("cluster_size: ",data.shape[1]//n_clusters)
            _, labels = kmeans.fit(data)
            # toc = time.time()
            # print(f"Balanced Kmeans time: {toc-tic:.4f}s")
        else:
            labels = np.linspace(0,data.shape[0]-1,data.shape[0]).astype(np.int32)
            
        data = data.cpu().numpy()
        unique_labels = np.unique(labels)
        for i,ul in enumerate(unique_labels):
            labels[labels==ul] = i
        unique_labels = np.unique(labels)
        numBlock = [np.sum(labels==ul).item() for ul in unique_labels]
        self.logger.info(f"\nnumBlock: {np.sum(np.array(numBlock))}\n")
        self.numBlock_each_MLP = np.array(numBlock)
        self.MLP_nums = len(unique_labels)
        self.n_blocks_perMLP = np.max(numBlock)
        rearr_data = np.zeros((self.n_blocks_perMLP,self.MLP_nums,n_v))
        # each t should have different num of labels
        MLP_idx_each_t = []
        last_label_idx = 0
        for t in range(self.num_timeStep):
            optimize_blocks_t = self.optimize_blocks[t]
            num_labels = int(torch.sum(optimize_blocks_t).item())
            MLP_idx_each_t.append(labels[last_label_idx:last_label_idx+num_labels])
            last_label_idx += num_labels

        for i,ul in tqdm(enumerate(unique_labels),total=len(unique_labels)): # 384,1000
            slice_data = data[labels==ul,:]
            rearr_data[:slice_data.shape[0],i,:] = slice_data
        latent_idx_each_t = _count_occurrences_2d(MLP_idx_each_t)
        # print("MLP_idx_each_t: ",MLP_idx_each_t)
        # print("latent_idx_each_t: ",latent_idx_each_t)
        return rearr_data,MLP_idx_each_t,latent_idx_each_t,unique_labels
        
  
    def getTrainLoader(self):
        #* No sampling is implemented in the code here
        d_v = self.dataset_varList[0]
        n_b,n_V = self.data[d_v].shape
        self.coords_each_MLP[d_v] = self.coords_each_MLP[d_v]#.astype(np.float32)
        
        coords = repeat(self.coords_each_MLP[d_v],'v c -> b v c',b=n_b) #(N,V,4)
        training_data_input = torch.FloatTensor(coords) # (N,V,4)
        training_data_output = torch.FloatTensor(self.data[d_v]) #(N,V)
        #TODO: Rethinking the code here, the code below might not necessary
        training_data_input = rearrange(training_data_input, 'n v c -> v n c') #rearange time 2e-4s
        training_data_output = rearrange(training_data_output, 'n v -> v n') #*DataLoader requires batchsize dim to be the first dimensions, this cost additional time to process
        # print("trainig data max&min :",training_data_output.max(),training_data_output.min()) #0.6784 -1.0
        # print("training data output shape: ",training_data_output.shape)
        # print("training data input shape :",training_data_input.shape)
        data = torch.utils.data.TensorDataset(training_data_input,training_data_output)
        train_loader = DataLoader(dataset=data,batch_size=self.batch_size,shuffle=True)

        return train_loader
    
    def getInfInps(self):
        d_v = self.dataset_varList[0]
        n_b,n_V = self.data[d_v].shape
        self.grid[d_v] = self.grid[d_v]#.astype(np.float32)
        coords = repeat(self.grid[d_v],'v c -> b v c',b=n_b) #(N,V,4)
        inf_coords = torch.FloatTensor(coords) # (N,V,4)
        inf_time = torch.arange(0,self.read_timesteps_len).unsqueeze(-1)
       
        return inf_coords,inf_time,self.target_v
    

if __name__ == "__main__":
    args = yaml_loader('./tasks/configs/MINER.yml')
    scale = 2
    dataset = MultiScaleScalarDataSet(args,scale=scale)
    dataset.ReadData(scale)
    