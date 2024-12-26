import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from skimage.transform import resize
import os
import numpy as np
import torch.optim as optim
import time
from model import *
from tqdm import tqdm
from skimage.io import imsave
from utils import *
from dataio import *
from logger import *
import time 
import timm
import torchvision
import wandb
import matplotlib.pyplot as plt
from einops import rearrange, repeat

NUM_CHUNKS = 10
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

class Solver():
    def __init__(self,model,dataset,args,logger,idx,scale=1.0):
        self.all_setting = args
        self.dataset = dataset
        self.model = model
        self.n_layers = self.model.n_layers
        self.logger = logger.getLogger("trainLogger")
        self.inflogger = logger.getLogger("easylogger")
        
        self.dataInfo = self.dataset.datasetInfo
        self.scale = scale
        self.mode = args['mode']
        
        self.model_setting = args['model_setting']
        self.train_setting = args['train_setting']
        self.num_epochs = self.train_setting['num_epochs'][idx]
        self.data_setting = args['data_setting']
        self.model_name = args['model_name']
        self.dataset_name = self.data_setting['dataset']
        self.log_rootDir = self.train_setting['log_root_dir']
        self.log_base_dir = args['log_base_dir']
        self.batch_size = self.data_setting['batch_size']
        self.enable_wandb = args['enable_wandb']
        self.quant_opt = self.train_setting['quant_opt']
        
        self.n_MLPs = self.dataset.MLP_nums
        self.all_MLPsID = torch.arange(0,self.n_MLPs).to(device)
        self.learn_MLPsID = torch.arange(0,self.n_MLPs).to(device)
        self.optimize_blocks = self.dataset.optimize_blocks # 1 means optimize, 0 means not optimize
        self.data_Min_Max = self.dataset.data_Min_Max # (t,n_MLPs,2): (t,n_MLPs,2)[:,0] is min, (t,n_MLPs,2)[:,1] is max
        self.lr = self.train_setting['lr']
        self.downScaleMethod = self.data_setting['downScaleMethod']
        
        self.model = model.to(device)
        self.weight_penalty = self.train_setting['weight_penalty']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_setting['lr'],betas=(0.9,0.999),weight_decay=self.weight_penalty)
        
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.75) # 0.75 2.3MB 49.64dB
        
        self.MSELoss = nn.MSELoss()
        self.BlockLoss = nn.MSELoss(reduction='none')
        #* -----PRUNING setting-----*#
        self.enable_prune = self.train_setting['enable_prune']
        if self.enable_prune:
            self.prune_block_prioritized_weight = self.train_setting['prune_block_prioritized_weight']
            self.parameters_to_prune = [(self.model.net[l].linear,'weight') for l in range(self.n_layers-1)]
            self.bias_to_prune = [(self.model.net[l].linear,'bias') for l in range(1,self.n_layers-1)]
            self.model._set_parameters_to_prune(self.parameters_to_prune,self.bias_to_prune)
            self.prune_interval = 75
            self.first_time_prune_epoch = 100
            self.n_prune = (self.num_epochs - self.first_time_prune_epoch - self.prune_interval)//self.prune_interval
            self.final_sparsity = self.train_setting['sparsity'][idx]
            self.final_sparsity_bias = self.train_setting['sparsity_bias'][idx]
            
            
            #* setting for time varying comrpession
            
            if self.dataInfo[self.dataset.dataset_varList[0]]['total_samples'] == 1:
                #* setting for 3D comrpession
                self.sparsity = [0.2,0.3]
                self.sparsity_bias = [0.2,0.3]
                self.prune_itera = [75,115]
            else:
                self.sparsity = [0.3,0.4,0.45,0.5]
                self.sparsity_bias = [0.3,0.4,0.45,0.5]
                self.prune_itera = [150,225,300,375]
            print("sparsity: ",self.sparsity)
            
            # self.sparsity_bias = sparcity_func(self.final_sparsity_bias,self.prune_interval,self.first_time_prune_epoch,self.n_prune,prune_i)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.75) # 0.75 2.3MB 49.64dB
            self.prune = lambda amount,importance_scores: prune.global_unstructured(
                self.parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
                importance_scores=importance_scores,
            )
            self.prune_bias = lambda amount,importance_scores: prune.global_unstructured(
                self.bias_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
                importance_scores=importance_scores,
            )
        #* -----QUANTIZE setting-----*#
        if self.quant_opt == 'cluster':
            self.n_bits = self.model_setting['n_bits'][idx]
            self.n_bits_bias = self.model_setting['n_bits_bias'][idx]
            self.n_bits_latent = self.model_setting['n_bits_latent']
            self.is_latent_quant = True if self.n_bits_latent != 0 else False
            self.parameters_to_quant = [(self.model.net[l].linear,'weight') for l in range(self.n_layers-1)]
            self.bias_to_quant = [(self.model.net[l].linear,'bias') for l in range(1,self.n_layers-1)]
            self.model._set_parameters_to_quant(self.parameters_to_quant,self.bias_to_quant)
            self.QAT_epochs = self.train_setting['QAT_epochs'][idx]
        self.itera=1
        #* -----LOG setting-----*#
        self.Log_path = os.path.join(self.log_base_dir,'Log')
        self.Results_path = os.path.join(self.log_base_dir,'Results')
        
        self.init_gradient_mask()
        
        if self.enable_wandb:
            wandb.login(key='xxx') # your wandb key if available
            self.wandbInit()
            
    def init_gradient_mask(self):
        numBlock_each_MLP = (self.dataset.numBlock_each_MLP).astype(np.int32)
        n_MLPs = self.dataset.MLP_nums
        self.gradient_mask_t = torch.zeros((n_MLPs,self.dataset.n_blocks_perMLP)).to(device)
        for i in range(n_MLPs):
            self.gradient_mask_t[i,0:numBlock_each_MLP[i]] = 1.0
            
    def train_one_iteration(self,t_coords,v,getBlockLoss=False):
        BlockLoss = None
        t_coords = t_coords.to(device)
        v = v.to(device)
        t_coords = rearrange(t_coords, 'v n c -> n v c') #about 1e-4s per call
        v = rearrange(v, 'v n -> n v')
        self.optimizer.zero_grad()
        coords, t = t_coords[...,1:], t_coords[...,0].long()
        v_pred = self.model(coords,self.learn_MLPsID,t)
        # v_pred = torch.squeeze(v_pred,(0,2)) 
        v_pred = torch.squeeze(v_pred) 
        v_pred.register_hook(lambda grad: grad * self.gradient_mask_t[:,t[0]]) # todo: need to double check whether this work
        loss = self.MSELoss(v_pred,v[self.learn_MLPsID,...].squeeze(0))
        loss.backward()
        self.optimizer.step()
        if getBlockLoss and len(self.learn_MLPsID)!=1:
            with torch.no_grad():
                v_pred = self.model(coords,self.all_MLPsID,t).squeeze()
                BlockLoss = self.BlockLoss(v_pred,v).detach().cpu().numpy()
                BlockLoss = np.mean(BlockLoss,axis=-1,keepdims=False)
        return loss.detach().item(), BlockLoss
   
    def train(self):
        train_loader = self.dataset.getTrainLoader()
        prune_idx = 0
        for itera in range(self.itera, self.num_epochs+1):
            accLoss = 0
            accBlockLoss = torch.zeros((self.n_MLPs))
            enable_BlockLoss = False
            if (self.enable_prune and (itera%self.prune_interval==0) and (itera >= self.first_time_prune_epoch)) or (self.enable_prune and itera == self.first_time_prune_epoch):# or (itera==1):
                enable_BlockLoss = True
            tic = time.time()
            for batch_idx, (coords,v) in enumerate(train_loader):
                loss, BlockLoss= self.train_one_iteration(coords, v, getBlockLoss=enable_BlockLoss)
                accBlockLoss += BlockLoss if BlockLoss is not None else 0.0
                accLoss += loss
            meanLoss = accLoss/(batch_idx+1)
            meanBlockLoss = accBlockLoss/(batch_idx+1)
            
            # if self.enable_prune and (itera%self.prune_interval==0 and itera >= self.first_time_prune_epoch):
            #     #visualize the block loss
            #     print(meanBlockLoss.shape) # (n_MLPs,)
            #     print(meanBlockLoss)
            #     # time.sleep(3)
            #* prune
            if self.enable_prune and self.sparsity[0] != 0:
                # if ((itera%self.prune_interval==0) and (itera >= self.first_time_prune_epoch)) or (itera == self.first_time_prune_epoch):
                if (itera in self.prune_itera):
                    improtance_map,bias_importance_map = self._get_importance_map(meanBlockLoss)
                    if prune_idx < len(self.sparsity):
                        # self.prune(0.4,improtance_map) #* prune 30% weights every 100 epochs # 47.51dB around 60%~65%
                        amount,bias_amount = self._get_amount_to_prune(self.sparsity[prune_idx],self.sparsity_bias[prune_idx])
                        self.prune(amount,improtance_map) #47.65dB around 60%~65%
                        self.prune_bias(bias_amount,bias_importance_map)
                        self.scheduler.step()
                    prune_idx += 1
                    print_sparsity(self.model)
                    
            if (itera%self.num_epochs==0) or (itera%300 == 0): # can not inference at the first epochs
                # exit()
                self.sample_inf(itera,self.scale) 
            #* lr update for stepLR
            
            toc = time.time()
            self.logger.info(f"Scale {self.scale} Epoch {itera} Loss: {meanLoss} Time: {toc-tic}, lr: {self.optimizer.param_groups[0]['lr']}")
            if self.enable_wandb:
                wandb.log({"loss": loss})
                
            # # visualize the weight distribution
            # if self.enable_prune and (itera%50==0) and (itera > self.first_time_prune_epoch):
            #     dummyModel = copy.deepcopy(self.model.to('cpu').state_dict())#* avoid change in-place
            #     # print("dummy models keys: ",dummyModel.keys())
            #     fig,axs = plt.subplots(1,self.n_layers)
            #     for l in range(self.n_layers):
            #         if ('net.'+str(l)+'.linear.weight_orig') not in dummyModel.keys():
            #             weight = dummyModel['net.'+str(l)+'.linear.weight'].detach().numpy()
            #             weight = weight[weight!=0]
            #             axs[l].hist(weight.flatten(),bins=300)
            #             axs[l].set_title(f"layer {l}")
            #         else:
            #             weight = dummyModel['net.'+str(l)+'.linear.weight_orig'].detach().numpy()*dummyModel['net.'+str(l)+'.linear.weight_mask'].detach().numpy()
            #             weight = weight[weight!=0]
            #             axs[l].hist(weight.flatten(),bins=300)
            #             axs[l].set_title(f"layer {l}")
            #     fig.savefig('./temp.png')
            #     fig.clf()
            #     self.model.to(device)
    
    def _weightCluster(self,model):
        model = model.to('cpu')
        #*<----- weight share layer by layer -----*>#
        model.apply_weight_share(n_bits=self.n_bits) #* NOTE: after apply weight share, remeber to reconstruct optimizer and scheduler to include new parameters
        model.apply_bias_share(n_bits=self.n_bits_bias)
        if self.is_latent_quant:
            model.apply_latent_share(n_bits=self.n_bits_latent)
        model = model.to(device)
        return model

    def applyWeightCluster(self):
        print("============START KMEANS QAT/PTQ!============")
        train_loader = self.dataset.getTrainLoader()
        self.model = self._weightCluster(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5,betas=(0.9,0.999))
        for itera in range(1, self.QAT_epochs):
            accLoss = 0
            tic = time.time()
            for batch_idx, (coords,v) in enumerate(train_loader):
                loss, BlockLoss = self.train_one_iteration(coords, v, getBlockLoss=False)
                accLoss += loss
            toc = time.time()
            meanLoss = accLoss/(batch_idx+1)
            self.logger.info(f"Scale {self.scale} Epoch {itera} Loss: {meanLoss} Time: {toc-tic}, lr: {self.optimizer.param_groups[0]['lr']}")

        self.sample_inf(itera,self.scale)
        print("============END KMEANS QAT/PTQ!============")
        
    def sample_inf(self,itera,scale):
        _,inf_time,target_v = self.dataset.getInfInps()
        d_v = self.dataset.dataset_varList[0]
        
        n_valid_blocks_t = self.dataset.n_MLPs_t
        MLP_idx_each_t = self.dataset.MLP_idx_each_t
        latent_idx_each_t = self.dataset.latent_idx_each_t
        
        volMesWidget = VolumeMetrics(GT_dirPath=parseDirfromP(self.dataInfo[d_v]['data_path']),\
                                     eval_dirPath=os.path.join(self.Results_path,self.dataset_name[0]+f"-scale1"))
        block_dims = self.dataset.block_dims
        n_voxels_per_block = int(np.prod(block_dims))
        self.model.eval()
        tic = time.time()
        with torch.no_grad():
            v_res_t = []
            idx = 0
            inf_block_dims = [bd for bd in block_dims]
            inf_coords_s = torch.FloatTensor(get_mgrid([inf_block_dims[0],inf_block_dims[1],inf_block_dims[2]],dim=3))
            Fold = self.dataset.Fold[d_v]
            block_start_idx = 0
            for t in tqdm(inf_time,disable=True):
                idx += 1
  
                if MLP_idx_each_t[0].shape[0] != 1:
                    MLP_indices = torch.LongTensor(MLP_idx_each_t[idx-1]).squeeze(0).to(device)
                    latent_labels_indices = torch.LongTensor(latent_idx_each_t[idx-1]).squeeze(0).to(device)     
                else:
                    MLP_indices = torch.LongTensor(MLP_idx_each_t[idx-1]).to(device)
                    latent_labels_indices = torch.LongTensor(latent_idx_each_t[idx-1]).to(device)
          
                inf_coords =  repeat(inf_coords_s,'v c -> b v c',b=MLP_indices.shape[0])

                time_col = torch.ones((inf_coords.shape[0],inf_coords.shape[1],1))*t
                inf_inps = torch.cat([time_col,inf_coords],dim=-1)
                n_MLPs,n_coords = inf_inps.shape[0],inf_inps.shape[1] #? check: n_MLPs should be equal to self.n_MLPs
                # print(inf_inps.shape)
                # print(MLP_indices.shape)
                # print(latent_labels_indices.shape)
                # exit()
                inf_inps_chunks = torch.split((inf_inps), 50, dim=0) #(n_blocks, batch_size, 4)
                MLP_indices_chunks = torch.split((MLP_indices), 50, dim=0)
                latent_labels_indices_chunks = torch.split((latent_labels_indices), 50, dim=0)
                # print(inf_inps_chunks[0].shape)
                # print(len(inf_inps_chunks))
                # exit()
                n_total_blocks = self.optimize_blocks.shape[1]
                n_valid_blocks = n_valid_blocks_t[idx-1]

                v_res = np.zeros((n_total_blocks,n_coords))
                v_pred_buffer = []
                # print(MLP_indices)
                if n_total_blocks == 1:
                    optimize_indices = self.optimize_blocks[t,...].nonzero(as_tuple=False).squeeze()
                else:
                    optimize_indices = self.optimize_blocks[t,...].squeeze().nonzero(as_tuple=False).squeeze()

                for chunck_idx, (inps,MLP_label,latent_label) in enumerate(zip(inf_inps_chunks,MLP_indices_chunks,latent_labels_indices_chunks)):
                    inps = inps.to(device)
                    # print(inps.shape)
                    # print(MLP_label.shape)
                    # print(latent_label.shape)
                    MLP_label = MLP_label.to(device)
                    latent_label = latent_label.to(device)
                    coords,timeIndex = inps[...,1:],t
                    # v_pred = self.model.inf(coords,MLP_indices,latent_labels_indices)
                    v_pred = self.model.inf(coords,MLP_label,latent_label)
                    v_pred_buffer += list(v_pred.squeeze(-1).detach().cpu().numpy())

                #* clamp the result to -1~1 then normalize back to original range
                #TODO: if do not use normalization, comment the following 3 lines except v_res = v_res.squeeze() 
                v_pred_buffer = np.array(v_pred_buffer,dtype=np.float32)#.squeeze(-1)
               
                blocks_max = self.data_Min_Max[d_v][block_start_idx:block_start_idx+n_valid_blocks,1,None].astype(np.float32)
                blocks_min = self.data_Min_Max[d_v][block_start_idx:block_start_idx+n_valid_blocks,0,None].astype(np.float32) #numpy can not boardcast 1d to 2d, so we expand the dim
                block_start_idx += n_valid_blocks
                v_pred_buffer = np.clip(v_pred_buffer,-1,1)
                v_pred_buffer = (v_pred_buffer/2+0.5)*(blocks_max-blocks_min)+blocks_min

                v_res[optimize_indices,:] = v_pred_buffer

                v_res = Fold(v_res,flatten_in=True,flatten_out=False).squeeze() # inference result for current scale
            
                D,H,W = v_res.shape
                v_res = np.asarray(v_res,dtype='<f')
                v_res_t.append(v_res)
                
                prev_result_dir = os.path.join(self.Results_path,d_v+f"-scale{int(scale*2)}")
                prev_result_paths = getFilePathsInDir(prev_result_dir,ext='.raw')
                if prev_result_paths != []: #if last scale inference dir exists, we add the residual to the result
                    last_scale_v = readDat(os.path.join(self.Results_path,d_v+f"-scale{int(scale*2)}",f"{self.model_name}-{self.dataset_name[0]}-{idx:04d}.raw"))
                    last_scale_v = last_scale_v.reshape((W//2,H//2,D//2)).transpose()
                    last_scale_v = resizeV(last_scale_v,scale_factor=2,flatten=False,keep_range=False)
                    v_res += last_scale_v
                if scale == 1:
                    v_res = np.clip(v_res,-1,1)
                v_res = v_res.flatten('F')
                # print("v_res norm max: ",v_res.max())
                # print("v_res norm min: ",v_res.min())
                #*save the result as results_path/dataset_var-scaleX/result.dat
                v_res.tofile(os.path.join(self.Results_path,d_v+f"-scale{scale}",f"{self.model_name}-{self.dataset_name[0]}-{idx:04d}.raw"),format='<f')

        toc = time.time()
        if self.enable_wandb:
            wandb.log({"Model size MB":print_model_size(self.model)})
        
        if scale == 1: #when scale is equal to 1, it's time to cal PSNR compare with GT
            meanPSNR,PSNR_ls = volMesWidget.getBatchPSNR()
            for t_idx,psnr in enumerate(PSNR_ls):
                self.logger.info(f"{self.model_name} Epoch {itera} timestep {t_idx} PSNR: {psnr}")
            self.logger.info(f"{self.model_name} Epoch {itera} Mean PSNR: {meanPSNR} time: {toc-tic}s")
            if self.enable_wandb:
                wandb.log({"psnr": meanPSNR, "inferece time": toc-tic})
        else:
            num_samples = len(target_v[d_v])
            psnr = []
            for t_idx in range(num_samples):
                target = target_v[d_v][t_idx]
                GT_range = target.max() - target.min()
                MSE = np.mean((v_res_t[t_idx].flatten('F') - target.flatten('F'))**2)
                _psnr = 20*np.log10(GT_range) - 10*np.log10(MSE)
                psnr.append(_psnr)
            psnr = np.array(psnr).mean()    
            print("psnr: ",psnr)
            if self.enable_wandb:
                wandb.log({"psnr": psnr, "inferece time": toc-tic})
        self.model.train()
    
    def _get_amount_to_prune(self,sparsity,sparsity_bias):
        #* weight
        block_mask = [
                getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
                for (module, name) in self.parameters_to_prune
            ]
        nelements = sum([m.nelement() for m in block_mask])
        nzeroelements = sum([m.nelement() - m.sum() for m in block_mask])
        n_elements_to_prune = int(sparsity * nelements - nzeroelements)
        #* bias
        bias_block_mask = [
                getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
                for (module,name) in self.bias_to_prune
        ]
        n_bias = sum([m.nelement() for m in bias_block_mask])
        nnz = sum([m.nelement() - m.sum() for m in bias_block_mask])
        n_bias_to_prune = int(sparsity_bias * n_bias - nnz)
        return n_elements_to_prune,n_bias_to_prune
    
    def _get_importance_map(self,block_loss):
        def _norm_map(m):
            return (m - m.min()) / (m.max() - m.min() + 1e-6)
        importance_map = [
            {}.get((module, name), getattr(module, name)).detach()
            for (module, name) in self.parameters_to_prune
        ]
        bias_importance_map = [
            {}.get((module, name), getattr(module, name)).detach()
            for (module, name) in self.bias_to_prune
        ]
        
        block_loss = block_loss.to(device)
        block_loss_params = [
            getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))*block_loss.unsqueeze(1).unsqueeze(2)
            for (module, name) in self.parameters_to_prune
        ] # get the mask for each layer and multiply with the block loss
        block_loss_per_MLP = torch.nn.utils.parameters_to_vector(block_loss_params) # convert block_loss_params to 1d vector for prune purpose
        
        bias_block_loss_params = [
            getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))*block_loss.unsqueeze(1).unsqueeze(2)
            for (module, name) in self.bias_to_prune
        ] # get the mask for each layer and multiply with the block loss
        bias_block_loss_per_MLP = torch.nn.utils.parameters_to_vector(bias_block_loss_params) # convert block_loss_params to 1d vector for prune purpose
        
        weight_importance = torch.nn.utils.parameters_to_vector(
        [
            {}.get((module, name), getattr(module, name)).detach()
            for (module, name) in self.parameters_to_prune
        ])
        weight_importance = torch.abs(weight_importance)
        
        bias_importance = torch.nn.utils.parameters_to_vector(
        [
            {}.get((module, name), getattr(module, name)).detach()
            for (module, name) in self.bias_to_prune
        ])
        bias_importance = torch.abs(bias_importance)
        # print(bias_importance.shape)
        # print(block_loss_per_MLP.shape)
        torch.nn.utils.vector_to_parameters(_norm_map(weight_importance)+self.prune_block_prioritized_weight*_norm_map(block_loss_per_MLP),importance_map)
        torch.nn.utils.vector_to_parameters(_norm_map(bias_importance)+self.prune_block_prioritized_weight*_norm_map(bias_block_loss_per_MLP),bias_importance_map)
        importance_score = {(module, name):importance_map[i] for i,(module, name) in zip(range(len(importance_map)),self.parameters_to_prune)}
        bias_importance_score = {(module, name):bias_importance_map[i] for i,(module, name) in zip(range(len(bias_importance_map)),self.bias_to_prune)}
        return importance_score, bias_importance_score
    

    def wandbInit(self):
        configs = flattenDict(self.all_setting)
        wandb.init(
            # set the wandb project where this run will be logged
            project=self.all_setting['project'],
            name=self.all_setting['name'],
            tags=self.all_setting['tags'],
            # track hyperparameters and run metadata
            config=configs
        )
    
    def setCheckPoint(self):
        # save model
        save_fileName = f"{self.model_name}"+f"-{self.scale}.pth"
        save_content = {
            'model_state_dict':self.model.state_dict(),
            'optimize_blocks':self.optimize_blocks
        }
        save_file_path = os.path.join(self.Log_path,save_fileName)
        torch.save(save_content,save_file_path)
        # save blocks meta data
        volDimInfo  = self.dataset.datasetInfo[self.dataset.dataset_varList[0]]['dim']
        volDimInfo = "_".join(list(map(str,volDimInfo))) # e.g. 256_256_256
        blockDimInfo = "_".join(list(map(str,self.dataset.block_dims))) # e.g. 16_16_16
        
        total_samples = self.dataset.total_samples
        timeCodeDim = self.model_setting['timeCodeDim']
        
        MetaFileName = f"Meta"+f"-{volDimInfo}"+f"-{blockDimInfo}"+f"-{self.scale}.npz"
        optimize_blocks_np = self.optimize_blocks.detach().cpu().numpy().astype(np.int0)
        # data_Min_Max_np = self.data_Min_Max[self.dataset.dataset_varList[0]]
        save_Meta_file_path = os.path.join(self.Log_path,MetaFileName)
        # np.savez_compressed(save_Meta_file_path,optimize_blocks=optimize_blocks_np,data_Min_Max_np=data_Min_Max_np)

    def resumeCheckPoint(self):
        lateset_model_path = getLatestModelPath(self.Log_path)
        saved_content = torch.load(lateset_model_path,map_location='cpu')
        self.model.load_state_dict(saved_content['model_state_dict'])
        self.optimize_blocks = saved_content['optimize_blocks']
    
    
        
def print_sparsity(model):
    n_layer = len(model.net)
    model = copy.deepcopy(model.state_dict()) #* avoid change in-place
    for k in model.keys():
        model[k] = model[k].cpu()
    global_Zero_elements = 0
    global_elements = 0
    for l in range(n_layer):
        weight_key = 'net.'+str(l)+'.linear.weight' if ('net.'+str(l)+'.linear.weight') in model.keys() else 'net.'+str(l)+'.linear.weight_mask'
        sparcity = 100. * float(torch.sum(model[weight_key] == 0))/float(model[weight_key].nelement())
        n_elements = float(model[weight_key].nelement())
        n_zero = float(torch.sum(model[weight_key] == 0))
        print(f"Sparsity in {l} layer.weight: {sparcity:.2f}%, n_ele: {n_elements}, n_zero: {n_zero}")
        global_elements += model[weight_key].nelement()
        global_Zero_elements += torch.sum(model[weight_key] == 0)
    print(
        "Global Sparsity: {:.2f}%".format(
            100. * float(global_Zero_elements)
            / float(global_elements)
        )
    )
    

