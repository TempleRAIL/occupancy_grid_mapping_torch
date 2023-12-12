# occupancy_grid_mapping_torch

## Introduction:
A GPU-accelerated and parallelized occupancy grid mapping algorithm that parallelizes the independent cell state update operations, written in pytorch.
More details can be found in our paper ["Stochastic Occupancy Grid Map Prediction in Dynamic Scenes"](https://openreview.net/pdf?id=fSmkKmWM5Ry)([arXiv](https://arxiv.org/abs/2210.08577)) in 7th Annual Conference on Robot Learning (CoRL) 2023. 

**Main code: dataset_gmapping_node.py**
* Input: n timesteps lidar measurements, robot poses, velocites (calculate the required coordinate reference frame) 
* Output: local occupancy grid map

## OGM-Datasets
The related datasets can be found at: https://doi.org/10.5281/zenodo.7051560. 
There are three different datasets collected by three different robot models (i.e. Turtlebot2, Jackal, Spot).
* 1.OGM-Turtlebot2: collected by a simulated Turtlebot2 with a maximum speed of 0.8 m/s navigates around a lobby Gazebo environment with 34 moving pedestrians using random start points and goal points
* 2.OGM-Jackal: extracted from two sub-datasets of the socially compliant navigation dataset (SCAND), which was collected by the Jackal robot with a maximum speed of 2.0 m/s at the outdoor environment of the UT Austin
* 3.OGM-Spot: extracted from two sub-datasets of the socially compliant navigation dataset (SCAND), which was collected by the Spot robot with a maximum speed of 1.6 m/s at the Union Building of the UT Austin

## Requirements:
* Python 3.7
* torch 1.7.1

##  Usage:
* Download OGM-datasets from https://doi.org/10.5281/zenodo.7051560 and decompress them to the home directory:
```Bash
cd ~
tar -zvxf OGM-datasets.tar.gz
```
* Mapping: note, modify the dataset path 'pDev' in "dataset_gmapping_node.py" to your OGM-datasets storage directory
```Bash
git clone https://github.com/TempleRAIL/occupancy_grid_mapping_torch.git
cd ./occupancy_grid_mapping_torch 
python dataset_gmapping_node.py
```

## Citation
If you find this code helpful, please cite this paper: 
```
@inproceedings{xie2023stochastic,
  title={Stochastic Occupancy Grid Map Prediction in Dynamic Scenes},
  author={Xie, Zhanteng and Dames, Philip},
  booktitle={Conference on Robot Learning},
  pages={1686--1705},
  year={2023},
  organization={PMLR}
}

@article{xie2023stochastic,
  title={Stochastic Occupancy Grid Map Prediction in Dynamic Scenes},
  author={Xie, Zhanteng and Dames, Philip},
  journal={arXiv preprint arXiv:2210.08577},
  year={2022}
}

```
