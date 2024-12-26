import numpy as np
from .filesUtils import *
from sklearn.neighbors import NearestNeighbors
import struct
import lpips
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from skimage import img_as_float
from skimage.io import imread
import torch


#*--------------------------------------------------------------------------------------------------*#
#* FileName: metricUtils.py
#* Last Modified: 2023-07-10
#* Metrics for measuring the quality of the reconstructed results (2D, 3D or surface files), 
#* including PSNR, SSIM, LPIPS, etc.
#*--------------------------------------------------------------------------------------------------*#

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VolumeMetrics():
    def __init__(self, GT_dirPath=None, eval_dirPath=None, verbose=False):
        self.GT_dirPath   = GT_dirPath
        self.eval_dirPath = eval_dirPath
        self.verbose      = verbose
        self.PSNR         = []
    
    def getPSNRFromFile(self,GT_datFile,eval_datFile):
        datGT    = np.fromfile(GT_datFile,dtype='<f')
        datEval  = np.fromfile(eval_datFile,dtype='<f')
        GT_range = np.max(datGT) - np.min(datGT)
        MSE      = np.mean((datEval - datGT)**2)
        PSNR     = 20*np.log10(GT_range) - 10*np.log10(MSE)
        return PSNR
    
    def getBatchPSNR(self):
        GTfilePaths   = getFilePathsInDir(self.GT_dirPath)
        EvalfilePaths = getFilePathsInDir(self.eval_dirPath)
        GTLength      = len(EvalfilePaths)
        if(len(GTfilePaths)==0):
            raise ValueError(f"VolumeMetrics.getBatchPSNR: No files found in {self.GT_dirPath}")
        for index in tqdm(range(GTLength),desc="Calculating Metrics",disable=(not self.verbose)): 
            self.PSNR.append(self.getPSNRFromFile(GTfilePaths[index],EvalfilePaths[index]))
        MeanPSNR = np.array(self.PSNR).mean()
        return MeanPSNR, self.PSNR



class ImageMetrics():
    def __init__(self, GT_dirPath=None, eval_dirPath=None, verbose=False):
        self.GT_dirPath   = GT_dirPath
        self.eval_dirPath = eval_dirPath
        self.verbose      = verbose
        self.SSIM         = []
        self.LPIPS        = []
        self.PSNR         = []
        self.lossfunc     = lpips.LPIPS(net='alex',version='0.1').to(device)
        
    
    def readImgFile(self,imgFilePath,metrics):
        if metrics in ["SSIM","PSNR"]:
            img        = img_as_float(imread(imgFilePath))
            img        = img.transpose(2,0,1)
            img_np     = np.asarray(img)
            img_np     = np.expand_dims(img_np,axis=0)
            img_tensor = torch.FloatTensor(img_np)
            return img_tensor
        elif metrics == "LPIPS":
            return lpips.im2tensor(lpips.load_image(imgFilePath))
        else:
            raise ValueError(f"ImageMetrics.readImgFile: Metrics {metrics} not defined")
    
    def getSSIMFromFile(self,gtImgFilePath, evalImgFilePath):
        imgGT   = self.readImgFile(gtImgFilePath,metrics="SSIM")
        imgEval = self.readImgFile(evalImgFilePath,metrics="SSIM")
        res     = ms_ssim(imgGT,imgEval,data_range=1,size_average=False)
        return res.item()

    def getLPIPSFromFile(self,gtImgFilePath, evalImgFilePath):
        imgGT   = self.readImgFile(gtImgFilePath, metrics="LPIPS")
        imgEval = self.readImgFile(evalImgFilePath, metrics="LPIPS")
        imgGT   = imgGT.to(device)
        imgEval = imgEval.to(device)
        res     = self.lossfunc.forward(imgGT, imgEval)
        return res.item()
    
    def getPSNRFromFile(self,gtImgFilePath, evalImgFilePath):
        imgGT   = self.readImgFile(gtImgFilePath, metrics="PSNR")
        imgEval = self.readImgFile(evalImgFilePath, metrics="PSNR")
        res = 10 * torch.log10(1 / torch.mean((imgGT - imgEval) ** 2))
        return res.item()
    
    def getBatchSSIM(self):
        GTfilePaths   = getFilePathsInDir(self.GT_dirPath)
        EvalfilePaths = getFilePathsInDir(self.eval_dirPath)
        GTLength      = len(EvalfilePaths)
        for index in tqdm(range(GTLength),desc="Calculating SSIM Metrics",disable=(not self.verbose)): 
            self.SSIM.append(self.getSSIMFromFile(GTfilePaths[index],EvalfilePaths[index]))
        MeanSSIM = np.array(self.SSIM).mean()
        return MeanSSIM, self.SSIM
    
    def getBatchLPIPS(self):
        GTfilePaths   = getFilePathsInDir(self.GT_dirPath)
        EvalfilePaths = getFilePathsInDir(self.eval_dirPath)
        GTLength      = len(EvalfilePaths)
        for index in tqdm(range(GTLength),desc="Calculating LPIPS Metrics",disable=(not self.verbose)): 
            self.LPIPS.append(self.getLPIPSFromFile(GTfilePaths[index],EvalfilePaths[index]))
        MeanLPIPS = np.array(self.LPIPS).mean()
        return MeanLPIPS, self.LPIPS

    def getBatchPSNR(self):
        GTfilePaths   = getFilePathsInDir(self.GT_dirPath)
        EvalfilePaths = getFilePathsInDir(self.eval_dirPath)
        GTLength      = len(EvalfilePaths)
        for index in tqdm(range(GTLength),desc="Calculating PSNR Metrics",disable=(not self.verbose)): 
            self.PSNR.append(self.getPSNRFromFile(GTfilePaths[index],EvalfilePaths[index]))
        MeanLPIPS = np.array(self.PSNR).mean()
        return MeanLPIPS, self.PSNR


class SurfaceMetrics():
    def __init__(self, GT_dirPath=None, eval_dirPath=None, verbose=False):
        self.GT_dirPath   = GT_dirPath
        self.eval_dirPath = eval_dirPath
        self.verbose      = verbose
        self.CD           = []
    
    def chamfer_distance(self, x, y, metric='l2', direction='bi'): 
    
        if direction=='y_to_x':
            x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
            min_y_to_x = x_nn.kneighbors(y)[0]
            chamfer_dist = np.mean(min_y_to_x)
        elif direction=='x_to_y':
            y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
            min_x_to_y = y_nn.kneighbors(x)[0]
            chamfer_dist = np.mean(min_x_to_y)
        elif direction=='bi':
            x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
            min_y_to_x = x_nn.kneighbors(y)[0]
            y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
            min_x_to_y = y_nn.kneighbors(x)[0]
            chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
        else:
            raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
            
        return chamfer_dist
    
    def readISOFile(self,filePath):
        with open(filePath,mode='rb') as file:
            num_vertices = int.from_bytes(file.read(4), byteorder='little')
            vertices = []
            for i in range(num_vertices):
                vertices.append(struct.unpack('ffff',file.read(16)))
            file.close()
            return vertices
    
    def getCDFromFile(self,GTfilePath,EvalfilePath):
        v_Eval = self.readISOFile(EvalfilePath)
        v_GT   = self.readISOFile(GTfilePath)
        return self.chamfer_distance(v_GT,v_Eval)
    
    def getBatchCD(self):
        GTfilePaths   = getFilePathsInDir(self.GT_dirPath)
        EvalfilePaths = getFilePathsInDir(self.eval_dirPath)
        GTLength      = len(EvalfilePaths)
        for index in tqdm(range(GTLength),desc="Calculating CD Metrics",disable=(not self.verbose)): 
            self.CD.append(self.getCDFromFile(GTfilePaths[index],EvalfilePaths[index]))
        MeanCD = np.array(self.CD).mean()
        return MeanCD, self.CD

def print_metrics_multivars(mean_dict,value_dict,varls,metics_name):
    total_val = 0
    for var in varls:
        total_val += mean_dict[var]
        for i in range(len(value_dict[var])):
            print(f"the {metics_name} for {var} at {i+1} time steps = {value_dict[var][i]}")
        print(f"Mean {metics_name} for {var} Value = {mean_dict[var]}")
    print(f"Total Mean {metics_name} Value = {total_val/len(varls)}")

def print_metrics(mean_val,value_ls,metics_name):
    for i in range(len(value_ls)):
        print(f"the {metics_name} at {i+1} time steps = {value_ls[i]}")
    print(f"Mean {metics_name} Value = {mean_val}")