# occupancy_grid_mapping_torch

## Introduction:
A GPU-accelerated and parallelized occupancy grid mapping algorithm that parallelizes the independent cell state update operations, written in pytorch.

#### Dataset: 
* Three occupancy grid map datasets collected by three different robot models can be found at: https://doi.org/10.5281/zenodo.7051560

#### Main code: dataset_gmapping_node.py  
* Input: n timesteps lidar measurements, robot poses, velocites (calculate the required coordinate reference frame) 
* Output: local occupancy grid map

####  Usage:
```
python dataset_gmapping_node.py
```

## Requirements:
* Python version at least 3.7
* Pytorch version at 1.7.1

## Citation
If you find this code helpful, please cite this paper: 
```
@article{xie2022stochastic,
  title={Stochastic Occupancy Grid Map Prediction in Dynamic Scenes},
  author={Xie, Zhanteng and Dames, Philip},
  year={2022}
}

```
