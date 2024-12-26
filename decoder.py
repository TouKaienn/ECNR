import argparse
from model import *
from dataio import *
from train import *
import os
import time
from torchsummary import summary
from utils import *
from tools import *
import torchsnooper
import inspect


class Decoder():
    def __init__(self,compFile,outDir='./out',verbose=True):
        #* ensure dirs then delete files in outDir
        print("============================Decoding============================")
        ensure_dirs(outDir)
        delFilesInDir(outDir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 1600
        self.outDir  =  outDir
        self.compFile = compFile
        self.file = open(self.compFile,'rb')
        self.header = self.file.read(24)
        self.nums_scale = struct.unpack('B',self.header[0:1])[0]
        self.init_feat = struct.unpack('B',self.header[1:2])[0]
        self.block_dims = struct.unpack('BBB',self.header[2:5])
        self.n_layers = struct.unpack('B',self.header[5:6])[0]
        self.timeCodeDim = struct.unpack('B',self.header[6:7])[0]
        self.omega_0 = struct.unpack('B',self.header[7:8])[0]
        self.dim = struct.unpack('III',self.header[8:20])
        self.total_samples = struct.unpack('I',self.header[20:24])[0]

        self.optimize_blocks = None
        self.data_min_max = None #just init theses vakues
        self.Fold = None #*init and remeber there is no overlapping
        # self.generateGrid(0)
        self.verbose = verbose
        if self.verbose:
            print("nums_scale: ",self.nums_scale)
            print("init_feat: ",self.init_feat)
            print("block_dims: ",self.block_dims)
            print("n_layers: ",self.n_layers)
            print("timeCodeDim: ",self.timeCodeDim)
            print("omega_0: ",self.omega_0)
            print("dim: ",self.dim)
            print("total_samples: ",self.total_samples)
        # exit()
        tic = time.time()
        self.decode()
        toc = time.time()
        print("Decoding time: ",toc-tic,"s")
        # n_MLPs = struct.unpack('I',self.file.read(1))[0]
        # print("test: ",n_MLPs)
    
    def decode(self):
        total_timesteps = [i for i in range(1,self.total_samples+1)]
        for scale_idx in range(self.nums_scale,-1,-1): #from coarse to fine
            scale = int(2**scale_idx)
            model,n_MLPs,MLP_idx_each_t,latent_idx_each_t,optimize_blocks,data_min_max = self.loadModel()
            cur_timeSteps = self.get_sampled_timeSteps(total_timesteps,scale)
            inf_coords,inf_time = self.generateGrid(n_MLPs,cur_timeSteps)
            #*<---Fold Function--->*# we do not support padding, you can implement it by yourself following the code in train.py
            stride = [int(d) for d in self.block_dims] #* parameter size of MLP might became 2 times bigger than the original size when stride have overlap part
            inf_output_shape = [d//scale for d in self.dim] 
            self.Fold = Fold3D(output_shape=inf_output_shape,kernel_size=self.block_dims,stride=stride)
            
            
            n_voxels_per_block = int(np.prod(self.block_dims))
        
            n_valid_blocks_t = [len(m) for m in MLP_idx_each_t]
            model = model.to(self.device)
            model.eval()
            
            prev_result_dir = self.outDir
            prev_result_paths = getFilePathsInDir(prev_result_dir,ext='.raw')
            with torch.no_grad():
                idx = 0
                inf_block_dims = [bd for bd in self.block_dims]
                inf_coords_s = torch.FloatTensor(get_mgrid([inf_block_dims[0],inf_block_dims[1],inf_block_dims[2]],dim=3))
                Fold = self.Fold
                block_start_idx = 0
                for t in tqdm(inf_time,disable=True):
                    idx += 1
                    MLP_indices = torch.LongTensor(MLP_idx_each_t[idx-1]).squeeze(0).to(self.device)
                    latent_labels_indices = torch.LongTensor(latent_idx_each_t[idx-1]).squeeze(0).to(self.device)       
                    inf_coords =  repeat(inf_coords_s,'v c -> b v c',b=MLP_indices.shape[0])

                    time_col = torch.ones((inf_coords.shape[0],inf_coords.shape[1],1))*t
                    inf_inps = torch.cat([time_col,inf_coords],dim=-1)
                    _,n_coords = inf_inps.shape[0],inf_inps.shape[1] #? check: n_MLPs should be equal to self.n_MLPs
                    inf_inps_chunks = torch.split(inf_inps, n_voxels_per_block+1, dim=1) #(n_blocks, batch_size, 4)
                    n_total_blocks = optimize_blocks.shape[1]
                    n_valid_blocks = n_valid_blocks_t[idx-1]
                    v_res = np.zeros((n_total_blocks,n_coords))
                    v_pred_buffer = []
                   
                    if n_total_blocks == 1:
                        optimize_indices = optimize_blocks[t,...].nonzero(as_tuple=False).squeeze()
                    else:
                        optimize_indices = optimize_blocks[t,...].squeeze().nonzero(as_tuple=False).squeeze()

                    for chunck_idx, inps in enumerate(inf_inps_chunks):
                        inps = inps.to(device)
                        coords,timeIndex = inps[...,1:],t
                        v_pred = model.inf(coords,MLP_indices,latent_labels_indices)
                        v_pred_buffer += list(v_pred.squeeze(-1).detach().cpu().numpy())

                    #* clamp the result to -1~1 then normalize back to original range
                    #TODO: if do not use normalization, comment the following 3 lines except v_res = v_res.squeeze()
                    v_pred_buffer = np.array(v_pred_buffer,dtype=np.float32)#.squeeze(-1)
                    
                    blocks_max = data_min_max[block_start_idx:block_start_idx+n_valid_blocks,1,None].astype(np.float32)
                    blocks_min = data_min_max[block_start_idx:block_start_idx+n_valid_blocks,0,None].astype(np.float32) #numpy can not boardcast 1d to 2d, so we expand the dim
                    block_start_idx += n_valid_blocks
                    v_pred_buffer = np.clip(v_pred_buffer,-1,1)
                    v_pred_buffer = (v_pred_buffer/2+0.5)*(blocks_max-blocks_min)+blocks_min

                    v_res[optimize_indices,:] = v_pred_buffer

                    v_res = Fold(v_res,flatten_in=True,flatten_out=False).squeeze() # inference result for current scale
                
                    D,H,W = v_res.shape
                    v_res = np.asarray(v_res,dtype='<f')
                    
                    if prev_result_paths != []: #if last scale inference dir exists, we add the residual to the result
                        # print(prev_result_paths)
                        # exit()
                        last_scale_v = readDat(os.path.join(self.outDir,f"vol-scale{scale*2}-{idx:04d}.raw"))
                        last_scale_v = last_scale_v.reshape((W//2,H//2,D//2)).transpose()
                        last_scale_v = resizeV(last_scale_v,scale_factor=2,flatten=False,keep_range=False)
                        v_res += last_scale_v
                        # print(f"remove vol-scale{scale*2}-{idx:04d}.raw")
                        # os.remove(os.path.join(self.outDir,f"vol-scale{scale*2}-{idx:04d}.raw"))
                    if scale == 1:
                        v_res = np.clip(v_res,-1,1)
                    v_res = v_res.flatten('F')
                    v_res.tofile(os.path.join(self.outDir,f"vol-scale{scale}-{idx:04d}.raw"),format='<f')
            
                #* space and time interpolation
            for p in prev_result_paths:
                os.remove(p)
            if scale != 1:
                linear_interpolation(dir_path=self.outDir, save_dir_path=self.outDir, total_samples=self.total_samples, scale=2*scale)
            
    def loadModel(self):
        global DEBUG_VAR
        n_MLPs = struct.unpack('Q',self.file.read(struct.calcsize('Q')))[0] #* decode n_MLPs
        n_blocks_perMLP = struct.unpack('Q',self.file.read(struct.calcsize('Q')))[0] #* decode n_Blocks
        timeCodeDim = struct.unpack('Q',self.file.read(struct.calcsize('Q')))[0] #* decode timeCodeDim
        init_feat = struct.unpack('Q',self.file.read(struct.calcsize('Q')))[0] #* decode init_feat
        n_layers = struct.unpack('Q',self.file.read(struct.calcsize('Q')))[0] #* decode n_layers
        n_bits = struct.unpack('Q',self.file.read(struct.calcsize('Q')))[0] #* decode n_bits
        n_bits_bias = struct.unpack('Q',self.file.read(struct.calcsize('Q')))[0] #* decode n_bits_bias
        n_bits_latent = struct.unpack('Q',self.file.read(struct.calcsize('Q')))[0] #* decode n_bits_latent
        optimize_shape_1 = struct.unpack('Q',self.file.read(struct.calcsize('Q')))[0] #* decode max_n_blocks_for_all_timesteps
        n_blocks_t_len = struct.unpack('Q',self.file.read(struct.calcsize('Q')))[0] #* decode n_blocks_t_len
        n_blocks_t = np.array(struct.unpack('Q'*n_blocks_t_len,self.file.read(struct.calcsize('Q')*n_blocks_t_len)))
        total_n_blocks = n_blocks_t.sum()
        data_min_max = np.array(struct.unpack('f'*total_n_blocks*2,self.file.read(struct.calcsize('f')*total_n_blocks*2))).reshape(total_n_blocks,2) #! Important
        # print(np.allclose(np.array(data_min_max).flatten(),DEBUG_VAR)) #*Correct!
        # exit()
        MLP_idx_each_t = []
        for i in range(len(n_blocks_t)):
            n_blocks = n_blocks_t[i]
            MLP_idx_1d_ls = list(struct.unpack('H'*n_blocks,self.file.read(struct.calcsize('H')*n_blocks)))
            MLP_idx_each_t.append(MLP_idx_1d_ls)
        # print("MLP_idx_each_t: ",MLP_idx_each_t)

        
        latent_idx_each_t = []
        for i in range(len(n_blocks_t)):
            n_blocks = n_blocks_t[i]
            latent_idx_1d_ls = list(struct.unpack('H'*n_blocks,self.file.read(struct.calcsize('H')*n_blocks)))
            latent_idx_each_t.append(latent_idx_1d_ls)

        optimize_blocks_1d_bytes = self.file.read((optimize_shape_1*n_blocks_t_len*1)//8)
        if (optimize_shape_1*n_blocks_t_len*1)%8 > 0:
                optimize_blocks_1d_bytes += self.file.read(1)

        optimize_blocks_1d_vec = one_bit_decode(optimize_blocks_1d_bytes,optimize_shape_1*n_blocks_t_len)
        # print("optimize_blocks_1d_vec: ",len(optimize_blocks_1d_vec))
        # print("optimize_blocks_1d_vec: ",optimize_blocks_1d_vec)
        # exit()
        optimize_blocks = torch.BoolTensor(optimize_blocks_1d_vec).view(n_blocks_t_len,optimize_shape_1)
        
        #todo: heck here flatten list
        MLP_idx_each_t_flatten = []
        for m in MLP_idx_each_t:
            MLP_idx_each_t_flatten += m
        
        MLP_idx_each_t_vec = np.array(MLP_idx_each_t_flatten).astype(np.int32).flatten()
        
        unique_labels = np.unique(MLP_idx_each_t_vec)
        # get latent mask (ignore dummy latent code)
        numBlock_each_MLP = np.array([np.sum(MLP_idx_each_t_vec==ul).item() for ul in unique_labels])
        MLP_nums = len(unique_labels)
        n_blocks_perMLP = np.max(numBlock_each_MLP)
        gradient_mask_t = torch.zeros((MLP_nums,n_blocks_perMLP))
        
        for i in range(n_MLPs):
            gradient_mask_t[i,0:numBlock_each_MLP[i]] = 1.0
        gradient_mask_t = gradient_mask_t.long()
        valid_latent_vec_len = int(gradient_mask_t.sum().item())
        
      
     
        model = ECNR(n_MLPs=n_MLPs,\
                    in_features   = 3,         \
                    n_blocks_perMLP  = n_blocks_perMLP,        \
                    timeCodeDim   = timeCodeDim,        \
                    out_features  = 1,         \
                    init_features = init_feat, \
                    n_layers      = n_layers,  omega_0 = self.omega_0)

        
        state_dict_buffer = model.state_dict() #* save everything into this buffer

        #* decode time latent code (no quant)
        latent_vec = torch.nn.utils.parameters_to_vector(state_dict_buffer['LatentTable']) # first stretch the tensor to a vector
        latent_vec = np.array(struct.unpack('f'*valid_latent_vec_len*timeCodeDim,self.file.read(struct.calcsize('f')*valid_latent_vec_len*timeCodeDim))) # then read the vector from file and reshape it to a tensor
        latent_vec_buffer = np.zeros((n_MLPs,n_blocks_perMLP,timeCodeDim)).astype(np.float32)
        latent_vec_buffer[gradient_mask_t==1.0,:] = latent_vec.reshape((-1,timeCodeDim))
        # print(torch.allclose(torch.tensor(latent_vec_buffer[gradient_mask_t==1.0,:]),torch.tensor(DEBUG_VAR))) #*correct!
        # exit()
        # model.LatentTable = torch.nn.Parameter(torch.FloatTensor(latent_vec_buffer))
        state_dict_buffer['LatentTable'] = torch.FloatTensor(latent_vec_buffer) # finally, assign the tensor to the model

        #*<---weight & bias--->*#
        #* mask (weight shape) -> labels (1d non zero index) -> centroids (2**n_bits 1d vec)
        for l in range(n_layers):
            weight_share = struct.unpack('?',self.file.read(struct.calcsize('?')))[0]
            pruned = struct.unpack('?',self.file.read(struct.calcsize('?')))[0]
            weight_key = f'net.{l}.linear.weight' if f'net.{l}.linear.weight' in state_dict_buffer.keys() else f'net.{l}.weight' 
        
            module = model.net[l].linear
            #* we only implement two cases, (share and prune) and (not share and not prune), implment other cases if you want
            if weight_share: # weight share
                if not pruned: # weight share and not pruned
                    raise ValueError("We do not support bias share and not pruned")
                else: # weight share and weight prune
                    module.is_weight_share = True
                    module.is_pruned = True
                    
                    weight_shape = state_dict_buffer[weight_key].shape
                    n_weight = state_dict_buffer[weight_key].view(-1).shape[0]#* corrent?
                    weight_mask_1d_byte = self.file.read(n_weight//8)
                    if n_weight%8 > 0:
                        weight_mask_1d_byte += self.file.read(1)
                    weight_mask_1d_ls = one_bit_decode(weight_mask_1d_byte,n_weight)
                    weight_mask = torch.BoolTensor(weight_mask_1d_ls).reshape(weight_shape)
                  
                    n_nonzero = weight_mask_1d_ls.count(1)
                    # print("nonzero: ",n_nonzero)
                    weight_labels_1d_byte = self.file.read((n_nonzero*n_bits)//8)
                    if (n_nonzero*n_bits)%8 > 0:
                        weight_labels_1d_byte += self.file.read(1)
                    
                    weight_labels = torch.LongTensor(bytes_to_bits_to_int(weight_labels_1d_byte,n_bits))
               
                    n_clusters = int(2**n_bits)
                    weight_centroids = torch.FloatTensor(struct.unpack('f'*n_clusters,self.file.read(struct.calcsize('f')*n_clusters)))
                   
                    module.centroids = weight_centroids.to(self.device)
                    module.labels = weight_labels.to(self.device)
                    module._weight_mask = weight_mask.to(self.device)
                    
                    # weight_buffer = torch.nn.utils.parameters_to_vector(state_dict_buffer[weight_key]) # first stretch the tensor to a vector
                    # weight_buffer.view(-1)[weight_mask.view(-1)[:]!=0] = weight_centroids[weight_labels]
                    # state_dict_buffer[weight_key].view(-1)[:] = weight_buffer
                    
                    # state_dict_buffer[weight_key].view(-1)[weight_mask.view(-1)[:]!=0] = weight_centroids[weight_labels]
            else: # no weight share
                if not pruned: #*OK
                    weight_buffer = torch.nn.utils.parameters_to_vector(state_dict_buffer[weight_key]) # first stretch the tensor to a vector
                    weight_buffer = torch.FloatTensor(struct.unpack('f'*weight_buffer.shape[0],self.file.read(struct.calcsize('f')*weight_buffer.shape[0]))) # then read the vector from file and reshape it to a tensor
                    # module.weight = weight_buffer.to(self.device)
                    state_dict_buffer[weight_key].view(-1)[:] = weight_buffer # finally, assign the tensor to the model
                else:
                    raise ValueError("We do not support bias not share and pruned")
            #*<---bias--->*#
            bias_share = struct.unpack('?',self.file.read(struct.calcsize('?')))[0]
            bias_prune = struct.unpack('?',self.file.read(struct.calcsize('?')))[0]
            bias_key = f'net.{l}.linear.bias' if f'net.{l}.linear.bias' in state_dict_buffer.keys() else f'net.{l}.bias' 
       
            if bias_share: # bias share
                if not bias_prune: # bias share and not pruned
                    raise ValueError("We do not support bias share and not pruned")
                else: # weight share and weight prune
                    module.is_bias_share = True
                    module.is_bias_pruned = True
                    bias_shape = state_dict_buffer[bias_key].shape
                    n_bias = state_dict_buffer[bias_key].view(-1).shape[0]#* corrent?
                    bias_mask_1d_byte = self.file.read(n_bias//8)
                    if n_bias%8 > 0:
                        bias_mask_1d_byte += self.file.read(1)
                    bias_mask_1d_ls = one_bit_decode(bias_mask_1d_byte,n_bias)
                    bias_mask = torch.BoolTensor(bias_mask_1d_ls).reshape(bias_shape)
                    n_nonzero = bias_mask_1d_ls.count(1)
                    # print("nonzero: ",n_nonzero)
                    bias_labels_1d_byte = self.file.read((n_nonzero*n_bits_bias)//8)
                    if (n_nonzero*n_bits_bias)%8 > 0:
                        bias_labels_1d_byte += self.file.read(1)
                    bias_labels = torch.LongTensor(bytes_to_bits_to_int(bias_labels_1d_byte,n_bits_bias))

                    n_clusters = int(2**n_bits_bias)
                    bias_centroids = torch.FloatTensor(struct.unpack('f'*n_clusters,self.file.read(struct.calcsize('f')*n_clusters)))
                    # print(bias_centroids.shape)
                    # print(torch.allclose(bias_centroids,DEBUG_VAR))
                    # exit()
                    # print(bias_key)
                    
                    module.centroids_bias = bias_centroids.to(self.device)
                    module.labels_bias = bias_labels.to(self.device)
                    module._bias_mask = bias_mask.to(self.device)
                    
                    # bias_buffer = torch.nn.utils.parameters_to_vector(state_dict_buffer[bias_key]) # first stretch the tensor to a vector
                    # bias_buffer.view(-1)[bias_mask.view(-1)[:] != 0] = bias_centroids[bias_labels]
                    # state_dict_buffer[bias_key].view(-1)[:] = bias_buffer
                    # state_dict_buffer[bias_key].view(-1)[bias_mask.view(-1)[:] != 0] = bias_centroids[bias_labels]
            else: # no bias share
                if not bias_prune: #*OK
                    bias_buffer = torch.nn.utils.parameters_to_vector(state_dict_buffer[bias_key]) # first stretch the tensor to a vector
                    bias_buffer = torch.FloatTensor(struct.unpack('f'*bias_buffer.shape[0],self.file.read(struct.calcsize('f')*bias_buffer.shape[0]))) # then read the vector from file and reshape it to a tensor
                    # module.bias = bias_buffer.to(self.device)
                    state_dict_buffer[bias_key].view(-1)[:] = bias_buffer # finally, assign the tensor to the model
                else:
                    raise ValueError("We do not support bias not share and pruned")
        model.load_state_dict(state_dict_buffer)
        return model,n_MLPs,MLP_idx_each_t,latent_idx_each_t,optimize_blocks,data_min_max
    
    def generateGrid(self,n_MLPs,timesteps):
        read_timesteps_len = len(timesteps)
        grid = get_mgrid([self.block_dims[0],self.block_dims[1],self.block_dims[2]],dim=3)
        coords = repeat(grid, 'v c -> b v c', b=n_MLPs)
        inf_coords = torch.FloatTensor(coords)
        inf_time = torch.arange(0,read_timesteps_len).unsqueeze(-1)
        return inf_coords,inf_time
    
    def get_sampled_timeSteps(self,timeSteps,scale):
        if scale == 1:
            return timeSteps
        final_element = timeSteps[-1]
        # print("time steps: ",timeSteps)
        while(scale != 1):
            read_timeSteps = []
            for idx in range(0,len(timeSteps),2):
                read_timeSteps.append(timeSteps[idx])
                if(len(timeSteps) - 4 == idx):
                    read_timeSteps.append(timeSteps[len(timeSteps)-1])
                    break
            timeSteps = read_timeSteps
            scale //= 2
            # print(f"self.read_timeSteps at scale {scale}: ",self.read_timeSteps)
        if(read_timeSteps[-1] != final_element):
            read_timeSteps.append(final_element)
        return read_timeSteps 
