from .filesUtils import *
from skimage.transform import resize
import torch.nn as nn
from scipy.ndimage import zoom
#*--------------------------------------------------------------------------------------------------*#
#* FileName: volumeUtils.py
#* Last Modified: 2023-07-22
#* Function tools for different voluem operation including: normalize, crop, etc. 
#* 
#*--------------------------------------------------------------------------------------------------*#

def BatchOn(dirPath,func,**kwargs):
    """This function can help you do a batch operation on all files in dirPath
    #! Note: batch operation do not reshape the data to 3D, so you need to reshape it by yourself in "func"
    Args:
        dirPath (path): dirPath for batch operation
        func (func): function for each volume
        kwargs (**dict): here you need to pass the parameters of func
    """
    batchRes = []
    paths = getFilePathsInDir(dirPath)
    for path in paths:
        v = readDat(path)
        batchRes.append(func(v,kwargs))
    return batchRes        

#*-----------------------------------volume operation----------------------------------------*#

def WrapTimeVols(dirPath,outFilePath,verbose=False):
    """concat volumes along time dimension and generate one 4D tensor file
       This function is mainly designed for tthresh method input
    Args:
        dirPath (_type_): _description_
        outFilePath (_type_): _description_
    """
    v_paths = getFilePathsInDir(dirPath,ext='.dat')
    VT = []
    for v_p in tqdm(v_paths,desc="wrapping volumes",disable=(not verbose)):
        VT += list(readDat(v_p))
    VT = np.asarray(VT,dtype="<f")
    saveDat(VT,outFilePath)

def FlattenTimeVols(FilePath,space_dims,outDir,name_prefix='tmp'):
    """Flatten 4d time volumes to separate 3d files, output these files into outDir
       Assign each output files with tmp name, so this function will clear the files in outDir beforehand
       This function is mainly designed for tthresh method output
    Args:
        FilePath (_type_): _description_
        space_dims (_type_): _description_
        outDir (_type_): _description_
    """
    ensure_dirs(outDir) #just confirm the outDir is exist
    delFilesInDir(outDir) #clear the outDir
    VT = readDat(FilePath)
    voxelNums = np.array(space_dims).prod()
    vols = np.split(VT,VT.shape[0]//voxelNums)
    for i in range(len(vols)):
        saveDat(vols[i],os.path.join(outDir,name_prefix+f'{i:04d}.dat'))
    

def resizeVol(vol,source_shape,target_shape,flatten=True,keep_range=True,mode='trilinear'):
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
    v_3d_resize = torch.nn.functional.interpolate(torch.from_numpy(vol[None,None,...]),size=target_shape,mode=mode).squeeze().numpy()
    if flatten:
        v_resize = v_3d_resize.flatten('F')
    else:
        v_resize = v_3d_resize
    if keep_range:
        v_resize = normalizeVol(vol=v_resize,outMin=v_min,outMax=v_max)
    return v_resize

def sampleVol(vol,source_shape,target_shape,flatten=True,keep_range=True):
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
    scale_x, scale_y, scale_z = source_shape[0]//target_shape[0],source_shape[1]//target_shape[1],source_shape[2]//target_shape[2]
    if len(vol.shape) >= 2:
        vol = vol.squeeze().transpose(2,1,0).reshape(-1)
    
    indices = []
    for z in range(0, source_shape[2],scale_z):
        for y in range(0, source_shape[1],scale_y):
            for x in range(0, source_shape[0],scale_x):
                index = ((z)*source_shape[1] + y)*source_shape[0] + x
                indices.append(index)
    v = vol[indices] #incex not support 3D 
    if flatten:
        v_sample = v
    else:
        v_sample = v.reshape(target_shape).transpose()
    if keep_range:
        v_sample = normalizeVol(vol=v_sample,outMin=v_min,outMax=v_max)
    # print(np.min(v_sample),np.max(v_sample))
    return v_sample


def normalizeVol(vol,outMin=0,outMax=1):
    """normalize vol to [outMin,outMax], input vol should be numpy array

    Args:
        vol (ndarry): _description_
        outMin (int, optional): _description_. Defaults to 0.
        outMax (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    vol_min,vol_max = np.min(vol),np.max(vol)
    zero_to_one_vol = (vol - vol_min)/(vol_max - vol_min)
    normalized_vol  = (outMax - outMin)*zero_to_one_vol + outMin
    return normalized_vol



#*--------------------------------------------------------------------------------------------*#



def NormalizeDataInDir(dirPath,outMin=0,outMax=1,outDir=None,verbose=True):
    """Normalize the volume scalar data in dir, then save the normalized data in outDir

    Args:
        dirPath (_type_): the path to load unnormalize data
        outMin (int, optional): min value after normalize. Defaults to 0.
        outMax (int, optional): max value after normalize. Defaults to 1.
        outDir (_type_, optional): the output dir which saved the normalized data. Defaults to "None" (which is dirPath).
    """
 
    primeDataPaths = getFilePathsInDir(dirPath)
    
    in_place_flag = False if outDir != None else True
    if outDir == None:
        outDir = dirPath
    else:
        ensure_dirs(outDir)
    
    for path in tqdm(primeDataPaths,disable=(not verbose),desc="normalizing data"):
        datFileName = os.path.split(path)[-1]
        dat         = readDat(path)
        dat_min,dat_max = np.min(dat),np.max(dat)
        zero_to_one_dat = (dat - dat_min)/(dat_max - dat_min)
        normalized_dat  = (outMax - outMin)*zero_to_one_dat + outMin
        normalized_dat = np.array(normalized_dat,dtype="<f")
        if in_place_flag:
            os.remove(path)
        saveDat(normalized_dat,os.path.join(outDir,datFileName))

def getVolPaddingSize(vol_dim,kernel_size,stride):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    padding = [0,0,0]

    for dim in range(3):
        n = 0
        while True:
            padding_d = (kernel_size[dim] + (stride[dim])*n - vol_dim[dim])/2
            if ((abs(int(padding_d)-padding_d)<1e-3) and (padding_d>=0)): #if we can get a padding_d which is int
                padding[dim] = int(padding_d)
                break
            elif((padding_d >= 0) and padding_d > vol_dim[dim]): #if padding_d is too large
                raise ValueError(f"getVolPaddingSize: There is no valid padding for vol dim {vol_dim}, kernel_size {kernel_size}, stride {stride}.\n \
                                    The reason is mainly caused by vol_dim which can not get a integer padding value.")
            n += 1
    return padding


class UnFold3D(nn.Module):
    """Support numpy or tensor, but did not support batch operation

    Args:
        nn (_type_): _description_
    """
    def __init__(self,kernel_size,padding=0,stride=1,dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation    
    
    def forward(self,inps,flatten_out=False):
        translate_toNumpy = False
        if type(inps) == np.ndarray:
            inps = torch.from_numpy(inps)
            translate_toNumpy = True
        if len(inps.shape) == 3:
            inps = inps.unsqueeze(0).unsqueeze(0)
        patches = extract_patches_3d(inps, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation)
        if flatten_out:
            patches = patches.squeeze()
            N = patches.shape[0]
            patches = patches.permute(0,3,2,1).reshape(N,-1)
        if translate_toNumpy:
            patches = patches.numpy()
        return patches

class Fold3D(nn.Module):
    """Support numpy or tensor, but did not support batch operation

    Args:
        nn (_type_): _description_
    """
    def __init__(self,output_shape,kernel_size,padding=0,stride=1,dilation=1,enable_weight_map=True, remove_padding=[0,0,0]):
        super().__init__()
        self.kernel_size = kernel_size
        self.output_shape = output_shape
        if len(output_shape) == 3:
            self.output_shape = (1,1,output_shape[0],output_shape[1],output_shape[2])
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.remove_padding = remove_padding
        weight_map = torch.ones(self.output_shape)
        weight_map_patches = extract_patches_3d(weight_map, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation)
        self.weight_map = combine_patches_3d(weight_map_patches, output_shape=self.output_shape, kernel_size=self.kernel_size,stride=self.stride, dilation=self.dilation, remove_padding=self.remove_padding)
        self.enable_weight_map = enable_weight_map
    
    def forward(self,inps,flatten_in=False,flatten_out=False):
        translate_toNumpy = False
        if type(inps) == np.ndarray:
            inps = torch.from_numpy(inps)
            translate_toNumpy = True
        if flatten_in:
            N = inps.shape[0]
            inps = inps.reshape(N,self.kernel_size[2],self.kernel_size[1],self.kernel_size[0]).unsqueeze(1)
            inps = inps.permute(0,1,4,3,2)
        if len(inps.shape) == 4:
            inps = inps.unsqueeze(1)
        res = combine_patches_3d(inps, kernel_size=self.kernel_size, output_shape=self.output_shape, padding=self.padding, stride=self.stride, dilation=self.dilation, remove_padding=self.remove_padding)       
        if self.enable_weight_map:
            res = res/self.weight_map #(B, C, D, H, W)
        if flatten_out:
            res = res.squeeze()
            res = res.permute(2,1,0).reshape(-1)
        if translate_toNumpy:
            res = res.numpy()
        return res
        

def extract_patches_3d(x, kernel_size, padding=0, stride=1, dilation=1):
    """_summary_
    https://stackoverflow.com/questions/68150248/how-to-extract-overlapping-patches-from-a-3d-volume-and-recreate-the-input-shape
    Args:
        x (tensor): tensor with shape (B, C, D, H, W)
        kernel_size (int or tuple): kernel size
        padding (int, optional): padding. Defaults to 0.
        stride (int or tuple, optional): stride. Defaults to 1.
        dilation (int, optional): dilation check this link https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding = 0, dim_stride = 1, dim_dilation = 1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    channels = x.shape[1]

    d_dim_in = x.shape[2]
    h_dim_in = x.shape[3]
    w_dim_in = x.shape[4]
    d_dim_out = get_dim_blocks(d_dim_in, kernel_size[0], padding[0], stride[0], dilation[0])
    h_dim_out = get_dim_blocks(h_dim_in, kernel_size[1], padding[1], stride[1], dilation[1])
    w_dim_out = get_dim_blocks(w_dim_in, kernel_size[2], padding[2], stride[2], dilation[2])
    # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)
    
    # (B, C, D, H, W)
    x = x.reshape(-1, channels, d_dim_in, h_dim_in * w_dim_in)                                                     
    # (B, C, D, H * W)

    x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[0], 1), padding=(padding[0], 0), stride=(stride[0], 1), dilation=(dilation[0], 1))                   
    # (B, C * kernel_size[0], d_dim_out * H * W)

    x = x.reshape(-1, channels * kernel_size[0] * d_dim_out, h_dim_in, w_dim_in)                                   
    # (B, C * kernel_size[0] * d_dim_out, H, W)

    x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[1], kernel_size[2]), padding=(padding[1], padding[2]), stride=(stride[1], stride[2]), dilation=(dilation[1], dilation[2]))        
    # (B, C * kernel_size[0] * d_dim_out * kernel_size[1] * kernel_size[2], h_dim_out, w_dim_out)

    x = x.reshape(-1, channels, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)  
    # (B, C, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)  

    x = x.permute(0, 1, 3, 6, 7, 2, 4, 5)
    # (B, C, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.contiguous().reshape(-1, channels, kernel_size[0], kernel_size[1], kernel_size[2])
    # (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

    return x

def combine_patches_3d(x, kernel_size, output_shape, padding=0, stride=1, dilation=1, remove_padding=[0,0,0]):
    """
    https://stackoverflow.com/questions/68150248/how-to-extract-overlapping-patches-from-a-3d-volume-and-recreate-the-input-shape
    When overlap happens, the output value is the sum of overlapping parts. To get the average value,
    you can use a augment weight map to get the average value. for example:
    >>> x = readDat(r"E:\VolumeData\supernova\Supernova_E0001.dat",toTensor=True)
    >>> x = x.reshape(256,256,256).transpose(2,0)
    >>> x = x.unsqueeze(0).unsqueeze(0)
    >>> weight_map = torch.ones_like(x)
    >>> res = extract_patches_3d(x, kernel_size=128,stride=128-64)
    >>> weight_map_patches = extract_patches_3d(weight_map, kernel_size=128,stride=128-64)
    >>> res = combine_patches_3d(res, output_shape=(1,1,256,256,256), kernel_size=128,stride=128-64)
    >>> weight_map_res = combine_patches_3d(weight_map_patches, output_shape=(1,1,256,256,256), kernel_size=128,stride=128-64)
    >>> torch.allclose(res,x) # output Flase
    >>> torch.allclose(res/weight_map_res,x) # output True
   
    Args:
        x (tensor): tensor with shape (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])
        kernel_size (int or tuple): kernel size
        output_shape (tuple): the output shape, if overlap is needed, overlap part will be the sum of overlapping parts
        padding (int, optional): _description_. Defaults to 0.
        stride (int, optional): _description_. Defaults to 1.
        dilation (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding = 0, dim_stride = 1, dim_dilation = 1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    channels = x.shape[1]
    d_dim_out, h_dim_out, w_dim_out = output_shape[2:]
    d_dim_in = get_dim_blocks(d_dim_out, kernel_size[0], padding[0], stride[0], dilation[0])
    h_dim_in = get_dim_blocks(h_dim_out, kernel_size[1], padding[1], stride[1], dilation[1])
    w_dim_in = get_dim_blocks(w_dim_out, kernel_size[2], padding[2], stride[2], dilation[2])
    # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

    x = x.reshape(-1, channels, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])
    # (B, C, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.permute(0, 1, 5, 2, 6, 7, 3, 4)
    # (B, C, kernel_size[0], d_dim_in, kernel_size[1], kernel_size[2], h_dim_in, w_dim_in)

    x = x.contiguous().reshape(-1, channels * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)
    # (B, C * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)

    x = torch.nn.functional.fold(x, output_size=(h_dim_out, w_dim_out), kernel_size=(kernel_size[1], kernel_size[2]), padding=(padding[1], padding[2]), stride=(stride[1], stride[2]), dilation=(dilation[1], dilation[2]))
    # (B, C * kernel_size[0] * d_dim_in, H, W)

    x = x.reshape(-1, channels * kernel_size[0], d_dim_in * h_dim_out * w_dim_out)
    # (B, C * kernel_size[0], d_dim_in * H * W)

    x = torch.nn.functional.fold(x, output_size=(d_dim_out, h_dim_out * w_dim_out), kernel_size=(kernel_size[0], 1), padding=(padding[0], 0), stride=(stride[0], 1), dilation=(dilation[0], 1))
    # (B, C, D, H * W)
    
    x = x.reshape(-1, channels, d_dim_out, h_dim_out, w_dim_out)
    # (B, C, D, H, W)
    for dim,remove_p in enumerate(remove_padding):
        if remove_p > 0:#* can not use weight_map to remove padding, it will OOM (CPU or GPU)
            if dim==0:
                x = x[:,:,remove_p:-remove_p,:,:]
            elif dim==1:
                x = x[:,:,:,remove_p:-remove_p,:]
            else:
                x = x[:,:,:,:,remove_p:-remove_p]
    return x

class UnFold4D(nn.Module):
    """Support numpy or tensor, but did not support batch operation

    Args:
        nn (_type_): _description_
    """
    def __init__(self,kernel_size,padding=0,stride=1,dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation    
    
    def forward(self,inps,flatten_out=False):
        translate_toNumpy = False
        if type(inps) == np.ndarray:
            inps = torch.from_numpy(inps)
            translate_toNumpy = True
        if len(inps.shape) == 3:
            inps = inps.unsqueeze(0).unsqueeze(0)
        patches = extract_patches_3d(inps, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation)
        if flatten_out:
            patches = patches.squeeze()
            N = patches.shape[0]
            patches = patches.permute(0,3,2,1).reshape(N,-1)
        if translate_toNumpy:
            patches = patches.numpy()
        return patches

class Fold4D(nn.Module):
    """Support numpy or tensor, but did not support batch operation

    Args:
        nn (_type_): _description_
    """
    def __init__(self,output_shape,kernel_size,padding=0,stride=1,dilation=1,enable_weight_map=True, remove_padding=[0,0,0]):
        super().__init__()
        self.kernel_size = kernel_size
        self.output_shape = output_shape
        if len(output_shape) == 3:
            self.output_shape = (1,1,output_shape[0],output_shape[1],output_shape[2])
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.remove_padding = remove_padding
        weight_map = torch.ones(self.output_shape)
        weight_map_patches = extract_patches_3d(weight_map, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation)
        self.weight_map = combine_patches_3d(weight_map_patches, output_shape=self.output_shape, kernel_size=self.kernel_size,stride=self.stride, dilation=self.dilation, remove_padding=self.remove_padding)
        self.enable_weight_map = enable_weight_map
    
    def forward(self,inps,flatten_in=False,flatten_out=False):
        translate_toNumpy = False
        if type(inps) == np.ndarray:
            inps = torch.from_numpy(inps)
            translate_toNumpy = True
        if flatten_in:
            N = inps.shape[0]
            inps = inps.reshape(N,self.kernel_size[2],self.kernel_size[1],self.kernel_size[0]).unsqueeze(1)
            inps = inps.permute(0,1,4,3,2)
        if len(inps.shape) == 4:
            inps = inps.unsqueeze(1)
        res = combine_patches_3d(inps, kernel_size=self.kernel_size, output_shape=self.output_shape, padding=self.padding, stride=self.stride, dilation=self.dilation, remove_padding=self.remove_padding)       
        if self.enable_weight_map:
            res = res/self.weight_map #(B, C, D, H, W)
        if flatten_out:
            res = res.squeeze()
            res = res.permute(2,1,0).reshape(-1)
        if translate_toNumpy:
            res = res.numpy()
        return res


def extract_patches_4ds(x, kernel_size, padding=0, stride=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding, padding, padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride, stride)

    channels = x.shape[1]

    x = torch.nn.functional.pad(x, padding)
    # (B, C, T, D, H, W)
    x = x.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1]).unfold(4, kernel_size[2], stride[2]).unfold(5, kernel_size[3], stride[3])
    # (B, C, t_dim_out, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2], kernel_size[3])
    x = x.contiguous().view(-1, channels, kernel_size[0], kernel_size[1], kernel_size[2], kernel_size[3])
    # (B * t_dim_out, d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2], kernel_size[3])
    return x

def extract_patches_4d(x, kernel_size, padding=0, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation, dilation)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding = 0, dim_stride = 1, dim_dilation = 1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    channels = x.shape[1]

    t_dim_in = x.shape[2]
    d_dim_in = x.shape[3]
    h_dim_in = x.shape[4]
    w_dim_in = x.shape[5]
    t_dim_out = get_dim_blocks(t_dim_in, kernel_size[0], padding[0], stride[0], dilation[0])
    d_dim_out = get_dim_blocks(d_dim_in, kernel_size[1], padding[1], stride[1], dilation[1])
    h_dim_out = get_dim_blocks(h_dim_in, kernel_size[2], padding[2], stride[2], dilation[2])
    w_dim_out = get_dim_blocks(w_dim_in, kernel_size[3], padding[3], stride[3], dilation[3])
    # print(t_dim_in, d_dim_in, h_dim_in, w_dim_in, t_dim_out, d_dim_out, h_dim_out, w_dim_out)
    
    # (B, C, T, D, H, W)
    x = x.view(-1, channels, t_dim_in, d_dim_in * h_dim_in * w_dim_in)                                                     
    # (B, C, T, D * H * W)

    x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[0], 1), padding=(padding[0], 0), stride=(stride[0], 1), dilation=(dilation[0], 1))
    # (B, C * kernel_size[0], t_dim_out * D * H * W)

    x = x.view(-1, channels * kernel_size[0] * t_dim_out, d_dim_in, h_dim_in * w_dim_in)                                   
    # (B, C * kernel_size[0] * t_dim_out, D, H * W)

    x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[1], 1), padding=(padding[1], 0), stride=(stride[1], 1), dilation=(dilation[1], 1))
    # (B, C * kernel_size[0] * t_dim_out * kernel_size[1], d_dim_out * H * W)

    x = x.view(-1, channels * kernel_size[0] * t_dim_out * kernel_size[1] * d_dim_out, h_dim_in, w_dim_in)
    # (B, C * kernel_size[0] * t_dim_out * kernel_size[1] * d_dim_out, H, W)

    x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[2], kernel_size[3]), padding=(padding[2], padding[3]), stride=(stride[2], stride[3]), dilation=(dilation[2], dilation[3]))        
    # (B, C * kernel_size[0] * t_dim_out * kernel_size[1] * d_dim_out * kernel_size[2] * kernel_size[3], h_dim_out * w_dim_out)

    x = x.view(-1, channels, kernel_size[0], t_dim_out, kernel_size[1], d_dim_out, kernel_size[2], kernel_size[3], h_dim_out, w_dim_out)
    # (B, C, kernel_size[0], t_dim_out, kernel_size[1], d_dim_out, kernel_size[2], kernel_size[3], h_dim_out, w_dim_out)

    x = x.permute(0, 1, 3, 5, 8, 9, 2, 4, 6, 7)
    # (B, C, t_dim_out, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2], kernel_size[3])

    x = x.contiguous().view(-1, channels, kernel_size[0], kernel_size[1], kernel_size[2], kernel_size[3])
    # (B * t_dim_out * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2], kernel_size[3])

    return x



def combine_patches_4d(x, kernel_size, output_shape, padding=0, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation, dilation)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding = 0, dim_stride = 1, dim_dilation = 1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    channels = x.shape[1]
    t_dim_out, d_dim_out, h_dim_out, w_dim_out = output_shape[2:]
    t_dim_in = get_dim_blocks(d_dim_out, kernel_size[0], padding[0], stride[0], dilation[0])
    d_dim_in = get_dim_blocks(d_dim_out, kernel_size[1], padding[1], stride[1], dilation[1])
    h_dim_in = get_dim_blocks(h_dim_out, kernel_size[2], padding[2], stride[2], dilation[2])
    w_dim_in = get_dim_blocks(w_dim_out, kernel_size[3], padding[3], stride[3], dilation[3])
    # print(t_dim_in, d_dim_in, h_dim_in, w_dim_in, t_dim_out, d_dim_out, h_dim_out, w_dim_out)

    x = x.view(-1, channels, t_dim_in, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2], kernel_size[3])
    # (B, C, t_dim_in, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2], kernel_size[3])

    x = x.permute(0, 1, 6, 2, 7, 3, 8, 9, 4, 5)
    # (B, C, kernel_size[0], t_dim_in, kernel_size[1], d_dim_in, kernel_size[2], kernel_size[3], h_dim_in, w_dim_in)

    x = x.contiguous().view(-1, channels * kernel_size[0] * t_dim_in * kernel_size[1] * d_dim_in * kernel_size[2] * kernel_size[3], h_dim_in * w_dim_in)
    # (B, C * kernel_size[0] * t_dim_in * kernel_size[1] * d_dim_in * kernel_size[2] * kernel_size[3], h_dim_in, w_dim_in)

    x = torch.nn.functional.fold(x, output_size=(h_dim_out, w_dim_out), kernel_size=(kernel_size[2], kernel_size[3]), padding=(padding[2], padding[3]), stride=(stride[2], stride[3]), dilation=(dilation[2], dilation[3]))
    # (B, C * kernel_size[0] * t_dim_in * kernel_size[1] * d_dim_in, H, W)

    x = x.view(-1, channels * kernel_size[0] * t_dim_in * kernel_size[1], d_dim_in * h_dim_out * w_dim_out)
    # (B, C * kernel_size[0] * t_dim_in * kernel_size[1], d_dim_in * H * W)

    x = torch.nn.functional.fold(x, output_size=(d_dim_out, h_dim_out * w_dim_out), kernel_size=(kernel_size[1], 1), padding=(padding[1], 0), stride=(stride[1], 1), dilation=(dilation[1], 1))
    # (B, C * kernel_size[0] * t_dim_in, D, H * W)

    x = x.view(-1, channels * kernel_size[0], t_dim_in * d_dim_out * h_dim_out * w_dim_out)
    # (B, C * kernel_size[0], t_dim_in * D * H * W)

    x = torch.nn.functional.fold(x, output_size=(t_dim_out, d_dim_out * h_dim_out * w_dim_out), kernel_size=(kernel_size[0], 1), padding=(padding[0], 0), stride=(stride[0], 1), dilation=(dilation[0], 1))
    # (B, C, T, D * H * W)
    
    x = x.view(-1, channels, t_dim_out, d_dim_out, h_dim_out, w_dim_out)
    # (B, C, T, D, H, W)

    return x