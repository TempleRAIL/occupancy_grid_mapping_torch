#!/usr/bin/env python
#
# file: /dataset_gmapping_node.py
#
# revision history: xzt
#  20220824 (TE): first version
#
# usage:
#
# This script is a sample code to read dataset and run the GPU-accelerated and parallelized occupancy grid mapping algorithm.
#------------------------------------------------------------------------------

import sys
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
import torch.nn as nn
from tqdm import tqdm
from local_occ_grid_map import LocalMap

# Init map parameters
P_prior = 0.5	# Prior occupancy probability
P_occ = 0.7	# Probability that cell is occupied with total confidence
P_free = 0.3	# Probability that cell is free with total confidence 
MAP_X_LIMIT = [0, 6.4] 
MAP_Y_LIMIT = [-3.2, 3.2]
RESOLUTION = 0.1 # Grid resolution in [m]'
TRESHOLD_P_OCC = 0.8

# torch device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function: get_data
#
# arguments: fp - file pointer
#            num_feats - the number of features in a sample
#
# returns: data - the signals/features
#          labels - the correct labels for them
#
# this method takes in a fp and returns the data and labels
NEW_LINE = "\n"
POINTS = 1080
IMG_SIZE = 64
SEQ_LEN = 10
class VaeTestDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, file_name):
        # initialize the data and labels
        # read the names of image data:
        self.scan_file_names = []
        self.pos_file_names = []
        self.vel_file_names = []
        # open train.txt or dev.txt:
        fp_scan = open(img_path+'/scans/'+file_name+'.txt', 'r')
        fp_pos = open(img_path+'/positions/'+file_name+'.txt', 'r')
        fp_vel = open(img_path+'/velocities/'+file_name+'.txt', 'r')
        # for each line of the file:
        for line in fp_scan.read().split(NEW_LINE):
            if('.npy' in line): 
                self.scan_file_names.append(img_path+'/scans/'+line)
        for line in fp_pos.read().split(NEW_LINE):
            if('.npy' in line): 
                self.pos_file_names.append(img_path+'/positions/'+line)
        for line in fp_vel.read().split(NEW_LINE):
            if('.npy' in line): 
                self.vel_file_names.append(img_path+'/velocities/'+line)
        # close txt file:
        fp_scan.close()
        fp_pos.close()
        fp_vel.close()
        self.length = len(self.scan_file_names)

        print("dataset length: ", self.length)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # get the index of start point:
        scans = np.zeros((SEQ_LEN+SEQ_LEN, POINTS))
        positions = np.zeros((SEQ_LEN+SEQ_LEN, 3))
        vels = np.zeros((SEQ_LEN+SEQ_LEN, 2))
        # get the index of start point:
        if(idx+(SEQ_LEN+SEQ_LEN) < self.length): # train1:
            idx_s = idx
        else:
            idx_s = idx - (SEQ_LEN+SEQ_LEN)

        for i in range(SEQ_LEN+SEQ_LEN):
            # get the scan data:
            scan_name = self.scan_file_names[idx_s+i]
            scan = np.load(scan_name)
            scans[i] = scan
            # get the scan_ur data:
            pos_name = self.pos_file_names[idx_s+i]
            pos = np.load(pos_name)
            positions[i] = pos
            # get the velocity data:
            vel_name = self.vel_file_names[idx_s+i]
            vel = np.load(vel_name)
            vels[i] = vel
        
        # initialize:
        scans[np.isnan(scans)] = 20.
        scans[np.isinf(scans)] = 20.
        scans[scans==30] = 20.

        positions[np.isnan(positions)] = 0.
        positions[np.isinf(positions)] = 0.

        vels[np.isnan(vels)] = 0.
        vels[np.isinf(vels)] = 0.

        # transfer to pytorch tensor:
        scan_tensor = torch.FloatTensor(scans)
        pose_tensor = torch.FloatTensor(positions)
        vel_tensor =  torch.FloatTensor(vels)

        data = {
                'scan': scan_tensor,
                'position': pose_tensor,
                'velocity': vel_tensor, 
                }

        return data


if __name__ == '__main__':
    # validation set and validation data loader
    BATCH_SIZE = 1
    pDev = "/home/xzt/OGM-datasets/OGM-Turtlebot2/val" # Change to your OGM-datasets storage directory
    eval_dataset = VaeTestDataset(pDev, 'val')
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, num_workers=2, \
                                                 shuffle=False, drop_last=True, pin_memory=True)
    # for each batch in increments of batch size:
    counter = 0
    # get the number of batches (ceiling of train_data/batch_size):
    num_batches = int(len(eval_dataset)/eval_dataloader.batch_size)
    for i, batch in tqdm(enumerate(eval_dataloader), total=num_batches):
        counter += 1
        # collect the samples as a batch: 10 timesteps
        scans = batch['scan']
        scans = scans.to(device)
        positions = batch['position']
        positions = positions.to(device)
        velocities = batch['velocity']
        velocities = velocities.to(device)
        
        # create occupancy grid maps: use 10 timesteps liar history as input, output the local occupancy grid map
        batch_size = scans.size(0)
        gridMap = LocalMap( X_lim = MAP_X_LIMIT, 
                            Y_lim = MAP_Y_LIMIT, 
                            resolution = RESOLUTION, 
                            p = P_prior,
                            size=[batch_size, SEQ_LEN],
                            device = device)
        # current position and velocities: 
        obs_pos_N = positions[:, SEQ_LEN-1]
        vel_N = velocities[:, SEQ_LEN-1]
        # the original pose of the local coordinate reference frame at t+n 
        T = 0 #int(t_pred)
        noise_std = [0, 0, 0]#[0.00111, 0.00112, 0.02319]
        pos_origin = gridMap.origin_pose_prediction(vel_N, obs_pos_N, T, noise_std)
        # robot positions:
        pos = positions[:,:SEQ_LEN]
        # Transform the robot past poses to the predicted reference frame.
        x_odom, y_odom, theta_odom =  gridMap.robot_coordinate_transform(pos, pos_origin)
        # Lidar measurements:
        distances = scans[:,:SEQ_LEN]
        # the angles of lidar scan: -135 ~ 135 degree
        angles = torch.linspace(-(135*np.pi/180), 135*np.pi/180, distances.shape[-1]).to(device)
        # Lidar measurements in X-Y plane: transform to the predicted robot reference frame
        distances_x, distances_y = gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
        # discretize to binary maps:
        input_binary_maps = gridMap.discretize(distances_x, distances_y)
        # occupancy map update:
        gridMap.update(x_odom, y_odom, distances_x, distances_y, P_free, P_occ)
        occ_grid_map = gridMap.to_prob_occ_map(TRESHOLD_P_OCC)

        # display the occupancy grid map:
        fig = plt.figure()
        a = fig.add_subplot(1,1,1)
        occ_map = occ_grid_map
        input_grid = make_grid(occ_map.detach().cpu())
        input_image = input_grid.permute(1, 2, 0)
        plt.imshow(input_image)
        plt.xticks([])
        plt.yticks([])
        plt.show()

