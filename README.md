# ECNR: Efficient Compressive Neural Representation of Time-Varying Volumetric Datasets
![alt text](https://github.com/TouKaienn/ECNR/blob/main/assets/ECNR-teaser.png)
## Description
This is the Pytorch implementation for ECNR: Efficient Compressive Neural Representation of Time-Varying Volumetric Datasets

## Installation
Create conda env:
```
git clone https://github.com/TouKaienn/STSR-INR.git
conda create --name ECNR python=3.9
conda activate ECNR
```
Install pytorch and other dependencies:
```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data Format
The volume at each time step is saved as a .raw file with the little-endian format. The data is stored in column-major order, that is, z-axis goes first, then y-axis, finally x-axis. You could download the static volume dataset supernova and time-varying volume dataset tangaroa with the link over here: [here](https://drive.google.com/drive/folders/1Hy2QZppXBZKN6JGW6V21AA9btg5ZK1dh)


Unzip the downloaded file ``Data.zip`` and put the data into the root dir, you could get a similar file structure like this:
```
.
├── assets
├── Data
│   ├── supernova
│   └── tangaroa
├── dataInfo
├── logger
├── ...
└── train.py
```


## Training and Inference
After Saving all your data in ./Data dir and then ensure ./dataInfo/localDataInfo.json includes all the necessary information for each volume data. Use the yaml file which contains all the hyper-parameters settings within ./configs dir to train or inference.


```
python3 main.py --config_path './configs/ionization_inf.yml'
```

To train from scratch:
```
python3 main.py --config_path './configs/ionization_train.yml'
```

After training or inference finished, you should be able to find the results in ./Exp dir.

## Citation
```
@inproceedings{tang2024ecnr,
  title={{ECNR}: Efficient Compressive Neural Representation of Time-Varying Volumetric Datasets},
  author={Tang, Kaiyuan and Wang, Chaoli},
  booktitle={Proceedings of IEEE Pacific Visualization Conference},
  pages={72-81},
  year={2024},
  doi={10.1109/PacificVis60374.2024.00017}
}
```
## Acknowledgements
This research was supported in part by the U.S. National Science Foundation through grants IIS-1955395, IIS-2101696, OAC2104158, and the U.S. Department of Energy through grant DESC0023145. The authors would like to thank the anonymous reviewers for their insightful comments.
