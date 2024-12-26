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
from decoder import Decoder
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#*=======Some vars used for debug=======*#
DEBUG_VAR = None
header_fileSize  = 0.0
weight_fileSize = 0.0
bias_fileSize  = 0.0
mapping_fileSize  = 0.0
mapping_maskSize = 0.0
mapping_labelsSize = 0.0
total_fileSize  = 0.0
latent_weight_fileSize = 0.0
COMP_PATH = None
#*=====================================*#
# @torchsnooper.snoop()
GT_dirPath = None #* this is for the convenience of evalDeCompressVol, might using parser in the future

def evalDeCompressVol(GT_dir,eval_dir):
    volMesWidget = VolumeMetrics(GT_dirPath=GT_dir,eval_dirPath=eval_dir)
    meanPSNR,PSNR_ls = volMesWidget.getBatchPSNR()
    for t_idx,psnr in enumerate(PSNR_ls):
        print(f"PSNR at time {t_idx+1}: ",psnr)
    print("mean PSNR: ",meanPSNR)

    
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


def main(opt):
    seed_everything()
    args,train_setting,model_setting,data_setting,logger = setup_Exp(opt)
    mode = args['mode']
    resultDir = os.path.join(args['log_base_dir'],'Results')
    block_dims = model_setting['block_dims']
    n_layers = model_setting['n_layers']
    timeCodeDim = model_setting['timeCodeDim']
    init_feat = model_setting['init']
    omega_0 = model_setting['omega_0']
    nums_scale = data_setting['nums_scale']
    n_bits = model_setting['n_bits']
    n_bits_bias = model_setting['n_bits_bias']
    n_bits_latent = model_setting['n_bits_latent']
    quant_opt = train_setting['quant_opt']
    ks = train_setting['ks']#[4,8,16,32]
    #*<---compression file--->*#
    outFile = os.path.join(args['log_base_dir'],opt.comp)
    file = open(outFile,'wb')
    #*<---global debug--->*#
    global header_fileSize 
    global weight_fileSize 
    global bias_fileSize 
    global mapping_labelsSize
    global mapping_maskSize
    global mapping_fileSize
    global latent_weight_fileSize
    global DEBUG_VAR   
    global COMP_PATH
    COMP_PATH = outFile
    #*<---header--->*#
    header_fileSize = file.write(struct.pack('B', nums_scale))
    
    header_fileSize += file.write(struct.pack('B', model_setting['init'][0])) #TODO: change here
    header_fileSize += file.write(struct.pack('BBB', block_dims[0][0],block_dims[0][1],block_dims[0][2])) #TODO: change here
    header_fileSize += file.write(struct.pack('B', n_layers[0])) #TODO: change here
    header_fileSize += file.write(struct.pack('B', model_setting['timeCodeDim']))
    header_fileSize += file.write(struct.pack('B', model_setting['omega_0']))
    
    datasetInfoPath = "./dataInfo/localDataInfo.json" if mode in ['debug', 'local_inf'] else './dataInfo/CRCDataInfo.json'
    datasetInfo = parseDatasetInfo(data_setting['dataset'],datasetInfoPath)
    global GT_dirPath
    GT_dirPath = parseDirfromP(datasetInfo[data_setting['dataset'][0]]['data_path']) #* for the convenience of evalDeCompressVol
    dim = datasetInfo[data_setting['dataset'][0]]['dim'] #* we assume all datasets have the same dims
    total_samples = datasetInfo[data_setting['dataset'][0]]['total_samples']
    
    header_fileSize += file.write(struct.pack('III', dim[0],dim[1],dim[2]))
    header_fileSize += file.write(struct.pack('I', total_samples))
    # header += file.write(struct.pack('B', 10))
    file.flush()
    file.close()
    # ks = [1,2,4,8]
    
    # ks = [1,2,4]
    for idx,i in enumerate(range(nums_scale,-1,-1)):
        scale = 2**i
        dataset = MultiScaleScalarDataSet(args,scale=scale,idx=idx,logger=logger)
        print("block dims: ",block_dims[idx])
        total_samples = dataset.total_samples
        if i == nums_scale:
            prev_resultRootDir = None
        else:
            prev_resultRootDir = resultDir
            print("prev result root dir: ",prev_resultRootDir)
        
        dataset.ReadData(scale,ks[idx],prev_resultRootDir)
        
        read_timesteps_len = dataset.read_timesteps_len

        n_MLPs = dataset.MLP_nums
        n_blocks_perMLP = dataset.n_blocks_perMLP
        optimize_blocks = dataset.optimize_blocks
        print("n_MLPs, n_blocks_perMLP: ",n_MLPs," ",n_blocks_perMLP)
        #*<---Create Model--->*#
        model = ECNR(n_MLPs=n_MLPs,\
                    in_features   = 3,         \
                    n_blocks_perMLP  = n_blocks_perMLP,        \
                    timeCodeDim   = timeCodeDim,        \
                    out_features  = 1,         \
                    init_features = init_feat[idx], \
                    n_layers      = n_layers[idx],  omega_0 = omega_0)
        print("Model Size (MB): ",print_model_size(model))
        #*<------Train&Prune->Quant------>*#
      
        t = Solver(model, dataset, args, logger, idx, scale)
        t.train()
        t.setCheckPoint()
        if quant_opt == 'cluster':
            t.applyWeightCluster()
        if scale != 1:
            # time linear interpolation
            dir_path = os.path.join(resultDir,f"{data_setting['dataset'][0]}-scale{scale}")
            linear_interpolation(dir_path=dir_path, save_dir_path=dir_path, total_samples=total_samples, scale=scale//2)
            
        #*<------ADD TO COMPRESS FILE------>*#
        file = open(outFile,'ab')
        #*<---save network setting--->*#
        header_fileSize += file.write(struct.pack('Q', n_MLPs))#! Save #* save n_MLPs
        header_fileSize += file.write(struct.pack('Q', n_blocks_perMLP))#! Save #* save n_blocks_perMLP
        header_fileSize += file.write(struct.pack('Q', timeCodeDim))#! Save #* save timeCodeDim
        header_fileSize += file.write(struct.pack('Q', init_feat[idx])) #! Save
        header_fileSize += file.write(struct.pack('Q', n_layers[idx]))#! Save
        header_fileSize += file.write(struct.pack('Q', n_bits[idx]))#! Save #* save n_bits
        header_fileSize += file.write(struct.pack('Q', n_bits_bias[idx]))#! Save #* save n_bits_bias
        header_fileSize += file.write(struct.pack('Q', n_bits_latent))#! Save  #* save n_bits_latent
        #* save data_min_max and optimize_blocks
        #MLP -> cluster num #latent -> cluster size
        # n_MLPs_t = dataset.n_MLPs_t
        latent_idx_each_t = dataset.latent_idx_each_t
        MLP_nums = dataset.MLP_nums 
        MLP_idx_each_t = dataset.MLP_idx_each_t
        optimize_blocks = t.optimize_blocks
        data_min_max = t.data_Min_Max[t.dataset.dataset_varList[0]].flatten().tolist() #* (n_total_blocks, 2)
        
        optimize_shape_1 = optimize_blocks.shape[1]
        n_blocks_t = [len(latent_idx_each_t[i]) for i in range(len(latent_idx_each_t))]
        # print(optimize_blocks.shape)
        # exit()
        #* save n_BLocks -> data_min_max (n_blocks, 2) -> optimize_blocks (t, n_Blocks)
        #*<---save blocks setting--->*#
        # print(max_n_blocks_for_all_timesteps)
        # print(len(data_min_max)) # 768
        # print(len(n_blocks_t)) # 6 -> timesteps
        # print(n_blocks_t)
        # print(optimize_blocks.shape) # (6, 64)
        # exit()
        header_fileSize += file.write(struct.pack('Q', optimize_shape_1))#! Save
        header_fileSize += file.write(struct.pack('Q', len(n_blocks_t))) # save n_blocks_t #! Save
        header_fileSize += file.write(struct.pack('Q'*len(n_blocks_t), *n_blocks_t)) # save n_blocks_t #! Save
        header_fileSize += file.write(struct.pack('f'*len(data_min_max), *data_min_max))#! Save
        # DEBUG_VAR = np.array(data_min_max)

        MLP_idx_each_t_vec = []
        for MLP_idx_t in MLP_idx_each_t:
            MLP_idx_each_t_vec += MLP_idx_t.tolist()
        
        # print("MLP_idx_each_t_vec: ",MLP_idx_each_t_vec)
        # print("len of MLP_idx_each_t_vec: ",len(MLP_idx_each_t_vec))
        
        
        header_fileSize += file.write(struct.pack('H'*len(MLP_idx_each_t_vec), *MLP_idx_each_t_vec)) # save n_blocks_t #! Save


        latent_idx_each_t_vec = []
        for latent_idx_t in latent_idx_each_t:
            latent_idx_each_t_vec += latent_idx_t
        # print("latent_idx_each_t_vec: ",latent_idx_each_t_vec)

     
        header_fileSize += file.write(struct.pack('H'*len(latent_idx_each_t_vec), *latent_idx_each_t_vec)) # save n_blocks_t #! Save
        
        optimize_blocks_vec = optimize_blocks.long().view(-1).tolist()
        
        # print("optimize_blocks_vec: ",len(optimize_blocks_vec)) 
        optimize_blocks_vec_bytes, _ = ints_to_bits_to_bytes(optimize_blocks_vec,1) # mask only contain 0 or 1
        header_fileSize += file.write(optimize_blocks_vec_bytes) # optimize_blocks #! Save

        model = t.model.to('cpu')
        state_dict = model.state_dict()
        net = model.net
        bias_to_prune = model.bias_to_prune
        
         
        #* Save Time Latent Code: avoid saving dummy latent code, remember that you could parse these values based on label_idx
        latent_vec  = getattr(model,'LatentTable').detach()
        is_latent_quanted = getattr(model,'is_latent_quanted')
        latent_vec_mask = t.gradient_mask_t.long().cpu()
        if not is_latent_quanted:
            latent_vec = latent_vec[latent_vec_mask==1.0,:].detach().numpy().flatten().tolist()
            latent_weight_fileSize += file.write(struct.pack('f'*len(latent_vec), *latent_vec)) #* save time latent code
        else:
            latent_centroids = getattr(model,'latent_centroids').cpu()
            latent_labels = getattr(model,'latent_labels').cpu()
            latent_vec.view(-1)[:] = latent_labels
            latent_vec = latent_vec[latent_vec_mask==1.0,:].detach().numpy().astype(np.int32).flatten().tolist()
            latent_centroids = latent_centroids.detach().numpy().flatten().tolist()
            latent_label_vec_bytes, is_leftover = ints_to_bits_to_bytes(latent_vec,n_bits_latent) # use n_bits to save labels
            latent_weight_fileSize += file.write(latent_label_vec_bytes) #* save time latent code
            latent_weight_fileSize += file.write(struct.pack('f'*len(latent_centroids), *latent_centroids)) #* save time latent code

        parameter_to_prune = t.parameters_to_prune if t.enable_prune else []
        bias_to_prune = t.bias_to_prune if t.enable_prune else []
        parameter_to_quant = t.parameters_to_quant if t.quant_opt == 'cluster'  else []
        bias_to_quant = t.bias_to_quant if t.quant_opt == 'cluster' else []
        
        for l in range(n_layers[idx]):  #*quant? -> pruned? + (weight mask (if pruned)) -> (labels if quant) + weight/centroids -> bias:       
            # there are multiple cases, just be patient...\(*^▽^*)/
            module = net[l].linear
            is_weight_pruned = (module,'weight') in parameter_to_prune
            is_weight_share = (module,'weight') in parameter_to_quant
            is_bias_pruned = (module,'bias') in bias_to_prune
            is_bias_share = (module,'bias') in bias_to_quant
            # print(l)
            # print("weight prune: ",is_weight_pruned)
            # print("weight share: ",is_weight_share)
            # print("bias prune: ",is_bias_pruned)
            # print("bias share: ",is_bias_share)
            #*<---save weight with quantization--->*#
            if is_weight_share: # if weight sharing
                header_fileSize += file.write(struct.pack('?', True)) #! Save # weight share?
                if not is_weight_pruned:
                    #* not pruned
                    header_fileSize += file.write(struct.pack('?', False)) #! Save # pruend? 
                    weight_labels_l = module.labels.cpu() # save n_bits
                    weight_centroids_l = module.centroids.cpu() # save 32 bits
                    
                    weight_labels_vec = weight_labels_l.view(-1).tolist()
                    weight_centroids_vec = weight_centroids_l.view(-1).tolist() # to 1d list
                    weight_labels_bytes, is_leftover = ints_to_bits_to_bytes(weight_labels_vec,n_bits[idx]) # use n_bits to save labels
                    mapping_labelsSize += file.write(weight_labels_bytes) # weight_labels #! Save
                    weight_fileSize += file.write(struct.pack('f'*len(weight_centroids_vec),*weight_centroids_vec)) # weight_centroids #! Save
                else: #* weight share and weight prune
                    header_fileSize += file.write(struct.pack('?', True)) #! Save # pruend? 
                    weight_centroids_l = module.centroids.cpu() # save 32 bits
                    weight_labels_l = module.labels.cpu() # save n_bits
                    weight_mask_l = module._weight_mask.cpu() # save 1 bit
                    
                    #* mask (weight shape) -> labels (1d non zero index) -> centroids (2**n_bits 1d vec) -> bias saving
                    weight_mask_vec = weight_mask_l.view(-1).tolist()  
                      
                   
                    weight_mask_vec_bytes, _ = ints_to_bits_to_bytes(weight_mask_vec,1) # mask only contain 0 or 1
                    mapping_maskSize += file.write(weight_mask_vec_bytes) # weight_mask #! Save
                    
                    weight_labels_vec = weight_labels_l.view(-1).tolist()
                    weight_labels_bytes, is_leftover = ints_to_bits_to_bytes(weight_labels_vec,n_bits[idx]) # use n_bits to save labels
                    mapping_labelsSize += file.write(weight_labels_bytes) # weight_labels #! Save
                    # DEBUG_VAR = np.array(weight_labels_vec)
                    weight_centroids_vec = weight_centroids_l.view(-1).tolist()
                    weight_fileSize += file.write(struct.pack('f'*len(weight_centroids_vec),*weight_centroids_vec)) # weight_centroids #! Save
                    # DEBUG_VAR = torch.tensor(weight_centroids_vec)
                # return
            else: # no weight sharing
                header_fileSize += file.write(struct.pack('?', False)) #! Save # weight share?
                if not is_weight_pruned:
                    #* not pruned
                    pruend = file.write(struct.pack('?', False))
                    # weight = state_dict.get(f'net.{l}.linear.weight') if f'net.{l}.linear.weight' in state_dict.keys() else state_dict.get(f'net.{l}.weight')
                    weight = module.weight.cpu() #* save weight
                    weight_vec = weight.view(-1).tolist()
                    weight_fileSize += file.write(struct.pack('f'*len(weight_vec), *weight_vec))
                    # print("weight_saving wo index: ",4*len(weight_vec)/1024,"KB")
                else:
                    pruned = file.write(struct.pack('?', True)) # pruned?
                    weight_orig = module.weight.cpu() #* weight
                    weight_mask = state_dict.get(f'net.{l}.linear.weight_mask') if f'net.{l}.linear.weight_mask' in state_dict.keys() else state_dict.get(f'net.{l}.weight_mask')
                    # weight_mask = module._weight_mask.cpu() #* mask
                    weight = weight_orig*weight_mask
                    weight_mask_vec = weight_mask.long().view(-1).tolist()
                    weight_nonzero_index = weight.view(-1).nonzero().view(-1)
                    
                    weight_mask_vec_bytes, _ = ints_to_bits_to_bytes(weight_mask_vec,1) # mask only contain 0 or 1
                    mapping_maskSize += file.write(weight_mask_vec_bytes) # weight_mask #! Save

                    # weight_mask_vec_str = ''.join(list(map(str,weight_mask_vec)))

                    # weight_mask_bytes, is_leftover = ints_to_bits_to_bytes([int(weight_mask_vec_str,1)],len(weight_mask_vec_str))
                    
                    weight_nonzero = weight.view(-1)[weight_nonzero_index].view(-1)
                    weight_vec = weight_nonzero.tolist()
                    # print("len of weight vec: ",len(weight_vec))
                    # print("len of str: ",len(weight_mask_vec_str))
                    
                    # print(f"weigh length at layer {l}: ",len(weight_vec))
                    # mapping_maskSize += file.write(weight_mask_bytes) # weight_index

                    weight_fileSize += file.write(struct.pack('f'*len(weight_vec), *weight_vec)) # weight
                    # print("weight_saving with index (before): ",2*4*len(weight_vec)/1024,"KB")
                    # print("weight_saving with index (after): ",4*len(weight_vec)/1024+len(weight_mask_bytes)/1024,"KB")
            # continue  
            #*<----------------------------------->*#
            if is_bias_share:
                header_fileSize += file.write(struct.pack('?', True)) #! Save # bias share?
                if not is_bias_pruned: 
                    #* not pruned
                    header_fileSize += file.write(struct.pack('?', False)) #! Save # pruend? 
                    bias_labels_l = module.labels_bias.cpu() # save n_bits
                    bias_centroids_l = module.centroids_bias.cpu() # save 32 bits
                    
                    bias_labels_vec = bias_labels_l.view(-1).tolist()
                    bias_centroids_vec = bias_centroids_l.view(-1).tolist() # to 1d list
                    bias_labels_bytes, is_leftover = ints_to_bits_to_bytes(bias_labels_vec,n_bits_bias[idx]) # use n_bits to save labels
                    mapping_labelsSize += file.write(bias_labels_bytes) # bias_labels #! Save
                    bias_fileSize += file.write(struct.pack('f'*len(bias_centroids_vec),*bias_centroids_vec)) # bias_centroids #! Save
                else: 
                    # print("OK are in")
                    header_fileSize += file.write(struct.pack('?', True)) #! Save # pruend? 
                    bias_centroids_l = module.centroids_bias.cpu() # save 32 bits
                    bias_labels_l = module.labels_bias.cpu() # save n_bits
                    bias_mask_l = module._bias_mask.cpu() # save 1 bit
                    #* mask (bias shape) -> labels (1d non zero index) -> centroids (2**n_bits 1d vec) -> bias saving
                    bias_mask_vec = bias_mask_l.view(-1).tolist()                
                    bias_mask_vec_bytes, _ = ints_to_bits_to_bytes(bias_mask_vec,1) # mask only contain 0 or 1
                    mapping_maskSize += file.write(bias_mask_vec_bytes) # bias_mask #! Save
                    
                    bias_labels_vec = bias_labels_l.view(-1).tolist()
                    bias_labels_bytes, is_leftover = ints_to_bits_to_bytes(bias_labels_vec,n_bits_bias[idx]) # use n_bits to save labels
                    mapping_labelsSize += file.write(bias_labels_bytes) # bias_labels #! Save

                    bias_centroids_vec = bias_centroids_l.view(-1).tolist()
                    bias_fileSize += file.write(struct.pack('f'*len(bias_centroids_vec),*bias_centroids_vec)) # bias_centroids #! Save
                    # DEBUG_VAR = torch.tensor(bias_centroids_l)
            else:
                header_fileSize += file.write(struct.pack('?', False)) #! Save # bias share?
                if not is_bias_pruned:
                    # save the bias
                    bias_pruned = file.write(struct.pack('?', False)) # bias pruned?
                    bias = state_dict.get(f'net.{l}.linear.bias') if f'net.{l}.linear.bias' in state_dict.keys() else state_dict.get(f'net.{l}.bias')
                    bias_vec = bias.view(-1).tolist()
                    bias_fileSize += file.write(struct.pack('f'*len(bias_vec), *bias_vec))
                else:
                    bias_pruned = file.write(struct.pack('?', True)) # bias pruned?
                    bias_mask = state_dict.get(f'net.{l}.linear.bias_mask') if f'net.{l}.linear.bias_mask' in state_dict.keys() else state_dict.get(f'net.{l}.bias_mask')
                    bias = state_dict.get(f'net.{l}.linear.bias_orig') if f'net.{l}.linear.bias_orig' in state_dict.keys() else state_dict.get(f'net.{l}.bias_orig')
                    bias = bias*bias_mask
                    bias_nnz_index = bias_mask.view(-1).nonzero().view(-1)
                    bias_nnz = bias.view(-1)[bias_nnz_index].view(-1)
                    bias_vec = bias_nnz.view(-1).tolist()
                    bias_mask_vec = bias_mask.long().view(-1).tolist()
                    bias_mask_bytes, is_leftover = ints_to_bits_to_bytes(bias_mask_vec,1)
                    mapping_maskSize += file.write(bias_mask_bytes) # bias_index
                    bias_fileSize += file.write(struct.pack('f'*len(bias_vec), *bias_vec))
       
        file.flush()
        file.close()
        fig,ax = plt.subplots()
        labels = ['header','weight','bias','mapping_mask','mapping_labels','latent']
        sizes = [header_fileSize,weight_fileSize,bias_fileSize,mapping_maskSize,mapping_labelsSize,latent_weight_fileSize]
        ax.pie(sizes,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
        fig.savefig(os.path.join(args['log_base_dir'],f'./scale-{nums_scale-i+1}-storage.png'))
        fig.clf()
    return 

if __name__ == "__main__":
    #combustion
    p = argparse.ArgumentParser()#multiVarCoord half-cylinder. argon  asteroid
    p.add_argument('--config_path', type=str, default='./tasks/configs/supernova.yml', help='The path of the config file') #diffusion
    p.add_argument('--out', type=str, default=None, help='The path of the out Dir') 
    p.add_argument('--comp', type=str, default='./comp', help='The path of the config file') 
    p.add_argument('--verbose', action='store_true', help='print debug info or not')
    opt = p.parse_args()
    print("Start Encoding...")
    s_tic = time.time()
    main(opt)
    e_toc = time.time()
    print("Total encoding time: ",(e_toc-s_tic)/(3600),"h")
    # outFile = opt.comp
    outFile = COMP_PATH
    outDir = "./out"
    Decoder(outFile,outDir=outDir)
    # #*<---eval--->*#
    print("FINAL PSNR: ")
    print(GT_dirPath)
    evalDeCompressVol(GT_dirPath,outDir)
    
