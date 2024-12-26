import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import *
import argparse

class Point():
    def __init__(self,x,y,z,idx=None,level=None):
        self.x = x
        self.y = y
        self.z = z
        self.idx = idx
        self.level = level
    
    def __repr__(self) -> str:
        return f"Point({self.x},{self.y},{self.z})"
        
    def setLevel(self,level):
        self.level = level
    
    def setIdx(self,idx):
        self.idx = idx

    def getPos(self):
        return [self.x,self.y,self.z]
    
class Mesh():
    def __init__(self,level,block_dims,vol_dims,idx_offset,optimize_blocks):
        self.optimize_blocks = optimize_blocks
        self.level = level
        self.points = []
        self.cells = np.array([]) #for the simplicity of boardcasting add offset
        self.vol_bbx = [[0,vol_dims[0]],[0,vol_dims[1]],[0,vol_dims[2]]]
        self.block_dims = block_dims
        self.cell_drawing_order = np.array([0,1,2,3,0,4,7,3,2,6,7,4,5,6,2,1,5,4]) #given 8 points, draw the cube
        self.idx_offset = idx_offset
        self.getPoints()
        self.n_points = len(self.points)
        self.getCells()

    def getOptimizeBlocks(self,model_path):
        saved_content = torch.load(model_path,map_location='cpu')
        optimize_blocks = list(map(int,saved_content['optimize_blocks'].tolist()))
        return optimize_blocks
        
        
    def getPointsFromCenter(self,center_point):
        corner_points = []
        front_left_bottom_p = Point(center_point.x-self.block_dims[0]/2,center_point.y-self.block_dims[1]/2,center_point.z-self.block_dims[2]/2)
        front_right_bottom_p = Point(center_point.x+self.block_dims[0]/2,center_point.y-self.block_dims[1]/2,center_point.z-self.block_dims[2]/2)
        back_left_bottom_p = Point(center_point.x-self.block_dims[0]/2,center_point.y+self.block_dims[1]/2,center_point.z-self.block_dims[2]/2)
        back_right_bottom_p = Point(center_point.x+self.block_dims[0]/2,center_point.y+self.block_dims[1]/2,center_point.z-self.block_dims[2]/2)
        front_left_top_p = Point(center_point.x-self.block_dims[0]/2,center_point.y-self.block_dims[1]/2,center_point.z+self.block_dims[2]/2)
        front_right_top_p = Point(center_point.x+self.block_dims[0]/2,center_point.y-self.block_dims[1]/2,center_point.z+self.block_dims[2]/2)
        back_left_top_p = Point(center_point.x-self.block_dims[0]/2,center_point.y+self.block_dims[1]/2,center_point.z+self.block_dims[2]/2)
        back_right_top_p = Point(center_point.x+self.block_dims[0]/2,center_point.y+self.block_dims[1]/2,center_point.z+self.block_dims[2]/2)
        corner_points = [front_left_bottom_p,front_right_bottom_p,back_right_bottom_p,back_left_bottom_p,front_left_top_p,front_right_top_p,back_right_top_p,back_left_top_p]

        return corner_points
    
        
    def getPoints(self):
        center_points = []
        block_idx = -1
        center_x_arr = np.arange(self.vol_bbx[0][0]+self.block_dims[0]/2,self.vol_bbx[0][1]-self.block_dims[0]/2+1,self.block_dims[0])
        center_y_arr = np.arange(self.vol_bbx[1][0]+self.block_dims[1]/2,self.vol_bbx[1][1]-self.block_dims[1]/2+1,self.block_dims[1])
        center_z_arr = np.arange(self.vol_bbx[2][0]+self.block_dims[2]/2,self.vol_bbx[2][1]-self.block_dims[2]/2+1,self.block_dims[2])
        for center_x in center_x_arr:
            for center_y in center_y_arr:
                for center_z in center_z_arr:
                    block_idx += 1
                    # print(block_idx)
                    if self.optimize_blocks is not None:
                        if self.optimize_blocks[block_idx] == 0:
                            continue
                        else:
                            center_points.append(Point(center_x,center_y,center_z))
                    else:
                        center_points.append(Point(center_x,center_y,center_z))
        # print(center_points)
        for c_p in center_points:
            corner_points = self.getPointsFromCenter(c_p)
            self.points.extend(corner_points)
       
    
    def getCells(self):
        Cells = np.zeros((self.n_points//8,len(self.cell_drawing_order)),dtype=np.int32)
        for i in range(self.n_points//8):
            Cells[i,:] = self.cell_drawing_order + i*8 + self.idx_offset
        self.cells = Cells
        # print(Cells)
    

def generateVTKFromPth(file_prefix,model_Dir,draw_levels=[0,1,2]):
    def _parse_model_paths(model_paths):
        # Parse the model path information to get the meta info
        n_levels = len(model_paths) # n_levels is equal to how many npz files you have in model_Dir
        vol_dim = None
        prime_block_dims  = [np.zeros(3) for i in range(n_levels)] #* different level can have different block_dims
        optimize_blocks = [None for i in range(n_levels)]
        n_timeStep = 0
        for idx,p in enumerate(model_paths): # read each model_path from fine to coarse
            fileName = parseFNfromP(p)
            metaInfo = fileName.split("-")
            if vol_dim is None:
                volInfo = list(map(int,metaInfo[1].split("_")))
                vol_dim = np.array(volInfo)
            # prime_block_dims[idx] = np.flip(np.array(list(map(int,metaInfo[2].split("_")))))
            prime_block_dims[idx] = np.array(list(map(int,metaInfo[2].split("_"))))
            optimize_blocks[idx] = np.load(model_paths[idx])['optimize_blocks']
            
        n_timeStep = optimize_blocks[0].shape[0]
        
        
        return n_levels,vol_dim,prime_block_dims,optimize_blocks,n_timeStep
    
    
    model_paths = getFilePathsInDir(model_Dir,".npz")
    n_levels,vol_dim,prime_block_dims,optimize_blocks,n_timeStep = _parse_model_paths(model_paths)
    # optimize_blocks[2] = np.ones_like(optimize_blocks[2]) # for debug
    if len(draw_levels) > n_levels:
        raise ValueError(f"draw_levels {draw_levels} should be less than or equal to {n_levels}")
    for t in range(1, n_timeStep+1):   # generate vtk files time-wisely
        file_name = file_prefix + f"{t:04d}.vtk"
        
        comment = "This is a test" 
        block_dims_at_level = {i:prime_block_dims[i]*(2**i) for i in range(n_levels)} # level fine (0) --> coarse (n_levels)
        Meshes_at_level = {i:None for i in range(n_levels)}
        previous_level_n_points = 0
        total_cell_nums = 0
        cell_nums_at_level = {i:0 for i in range(n_levels)}
        
        for l in range(n_levels-1,-1,-1):
            if l not in draw_levels:
                continue # skip some levels
            print(f"Generating vtk file for level {l} at time step {t} with optimized_bloks index: {(t-1)//(2**l)}")
            Meshes_at_level[l] = Mesh(l,block_dims_at_level[l],vol_dim,previous_level_n_points,optimize_blocks[l][(t-1)//(2**l)])
            previous_level_n_points += Meshes_at_level[l].n_points
            total_cell_nums += Meshes_at_level[l].cells.shape[0]
            cell_nums_at_level[l] = Meshes_at_level[l].cells.shape[0]
            # break
        total_num_Points = previous_level_n_points
        with open(file_name,'w+') as f:
            f.write("# vtk DataFile Version 3.0\n") # version
            f.write(comment+"\n")                   # comment
            f.write("ASCII\n")                      # coding method
            f.write("DATASET UNSTRUCTURED_GRID\n")  # dataset type
            # write points
            f.write(f"POINTS {total_num_Points} float\n")
            for i in range(n_levels-1,-1,-1): # coarse --> fine
                if i not in draw_levels:
                    continue # skip coarse level
                for p in Meshes_at_level[i].points:
                    f.write(f"{p.x} {p.y} {p.z}\n")
                
            # write cells
            f.write(f"CELLS {total_cell_nums} {total_cell_nums*(18+1)}\n") # Since we only use polydata, each row definitely be 18 points
            for i in range(n_levels-1,-1,-1):
                if i not in draw_levels:
                    continue # skip coarse level
                for c in Meshes_at_level[i].cells:
                    pointIdx_str = " ".join([str(idx) for idx in c])
                    write_str = "18 "+pointIdx_str
                    f.write(write_str+"\n")
                # break
            # write cell types        
            f.write(f"CELL_TYPES {total_cell_nums}\n")
            f.write("4\n"*total_cell_nums)
            
            # write cell data
            f.write(f"CELL_DATA {total_cell_nums}\n")
            f.write("SCALARS density float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for i in range(n_levels-1,-1,-1):
                if i not in draw_levels:
                    continue # skip coarse level
                f.write(f"{2-i} "*cell_nums_at_level[i])
                f.write("\n")
                # break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate VTK files to visualize the optimization process and blocks distribution')
    parser.add_argument('--model_Dir', type=str, default=r"/home/dullpigeon/Desktop/ECNR_result/TangaroaVTM/MetaFiles/Log/", help='The path of the Log directory which contain Metaxxx.npz files')
    parser.add_argument('--file_prefix', type=str, default="/home/dullpigeon/Desktop/ECNR_result/TangaroaVTM/MetaFiles/blocks/blocks", help='The prefix of the output vtk file name')

    args = parser.parse_args()
    
    model_Dir = args.model_Dir
    file_prefix = args.file_prefix
    file_prefix_Dir = parseDirfromP(file_prefix)
    ensure_dirs(file_prefix_Dir) # just ensure the dir of file_prefix exists
    
    generateVTKFromPth(file_prefix,model_Dir) # run the main function
