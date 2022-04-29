#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

import sys
from grid_map import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

P_prior = 0.5	# Prior occupancy probability
P_occ = 0.9	# Probability that cell is occupied with total confidence
P_free = 0.3	# Probability that cell is free with total confidence 

RESOLUTION = 0.075 # Grid resolution in [m]'


# function: get_data
#
# arguments: fp - file pointer
#            num_feats - the number of features in a sample
#
# returns: data - the signals/features
#          labels - the correct labels for them
#
# this method takes in a fp and returns the data and labels
SEED1 = 1337
NEW_LINE = "\n"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 10
class VaeTestDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, file_name):
        # initialize the data and labels
        self.length1 = 0
        # read the names of image data:
        self.scan_file_names = []
        self.pos_file_names = []
        self.vel_file_names = []
        # read the names of image data:
        self.img_file_names2 = []
        self.vel_file_names2 = []
        # open train.txt or dev.txt:
        fp_scan = open(img_path+'/scans/'+file_name+'.txt', 'r')
        fp_pos = open(img_path+'/odometry/'+file_name+'.txt', 'r')
        fp_vel = open(img_path+'/velocities/'+file_name+'.txt', 'r')
        # for each line of the file:
        for line in fp_scan.read().split(NEW_LINE):
            if('.npy' in line): 
                self.scan_file_names.append(img_path+'/scans/'+line)
        for line in fp_pos.read().split(NEW_LINE):
            if('.npy' in line): 
                self.pos_file_names.append(img_path+'/odometry/'+line)
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
        scans = np.zeros((SEQ_LEN+1, POINTS))
        positions = np.zeros((SEQ_LEN+1, 3))
        vels = np.zeros((SEQ_LEN+1, 2))
        # get the index of start point:
        if(idx+(SEQ_LEN+1) < self.length): # train1:
            idx_s = idx
        else:
            idx_s = idx - (SEQ_LEN+1)

        for i in range(SEQ_LEN+1):
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
        
        scans[np.isnan(scans)] = 0.
        scans[np.isinf(scans)] = 0.
        scans[scans==30] = 20.

        positions[np.isnan(positions)] = 0.
        positions[np.isinf(positions)] = 0.

        '''
        # Max-Min normalization:
        scan_max = 20
        scan_min = 0
        scans = (scans - scan_min) / (scan_max - scan_min)
        positions = (positions - scan_min) / (scan_max - scan_min)
        '''

        # transfer to pytorch tensor:
        scan_tensor = torch.FloatTensor(scans[:-1])
        #scan_mask_tensor = torch.FloatTensor(scans[-1])
        #scan_mask_tensor = scan_mask_tensor.reshape(POINTS)

        pose_tensor = torch.FloatTensor(positions[:-1])

        vel_tensor =  torch.FloatTensor(vels[:-1])

        data = {
                'scan': scan_tensor,
                'position': pose_tensor,
                'velocity': vel_tensor, 
                #'mask': scan_mask_tensor
                }

        return data

def lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom):
	"""
	Lidar measurements in X-Y plane
	"""
	distances_x = np.array([])
	distances_y = np.array([])

	for (dist, ang) in zip(distances, angles):
		distances_x = np.append(distances_x, x_odom + dist * np.cos(ang + theta_odom))
		distances_y = np.append(distances_y, y_odom + dist * np.sin(ang + theta_odom))

	return (distances_x, distances_y)

if __name__ == '__main__':
    # Init map parameters
    map_x_lim = [0,6] #[-10, 10]
    map_y_lim = [-3,3]#[-10, 10]

    # validation set and validation data loader
    BATCH_SIZE = 1
    pDev = "/home/xzt/vae_datasets/vae_dataset_v10/train"
    eval_dataset = VaeTestDataset(pDev, 'train')
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, num_workers=2, \
                                                 shuffle=True, drop_last=True, pin_memory=True)

    # for each batch in increments of batch size:
    counter = 0
    # get the number of batches (ceiling of train_data/batch_size):
    num_batches = int(len(eval_dataset)/eval_dataloader.batch_size)
    for i, batch in tqdm(enumerate(eval_dataloader), total=num_batches):
        counter += 1
        # collect the samples as a batch: 10 timesteps
        scans = batch['scan']
        positions = batch['position']
        velocities = batch['velocity']
        
        # Create grid map 
        gridMap = GridMap(X_lim = map_x_lim, 
                    Y_lim = map_y_lim, 
                    resolution = RESOLUTION, 
                    p = P_prior)

        # refrence frame: t = 10th timestep:
        pos_orign = positions[0, 9].detach().cpu().numpy()
        # Main loop
        # for BGR image of the grid map
        #X2 = []
        #Y2 = []
        for t in range(SEQ_LEN):
            scan = scans[0, t].detach().cpu().numpy()
            pos = positions[0, t].detach().cpu().numpy()
            vel = velocities[0, t].detach().cpu().numpy()

            # Lidar measurements
            distances = scan
            # the angles of lidar scan: -135 ~ 135 degree
            angles = np.linspace(-(135*np.pi/180), 135*np.pi/180, np.shape(distances)[0], endpoint = 'true')

            # Odometry measurements
            x_odom = pos[0] - pos_orign[0]
            y_odom = pos[1] - pos_orign[1]
            theta_odom = pos[2] - pos_orign[2]

            # Lidar measurements in X-Y plane
            distances_x, distances_y = lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)

            # x1 and y1 for Bresenham's algorithm
            x1, y1 = gridMap.discretize(x_odom, y_odom)
            
            # for BGR image of the grid map
            X2 = []
            Y2 = []
        

            for (dist_x, dist_y, dist) in zip(distances_x, distances_y, distances):

                # x2 and y2 for Bresenham's algorithm
                x2, y2 = gridMap.discretize(dist_x, dist_y)

                # draw a discrete line of free pixels, [robot position -> laser hit spot)
                for (x_bres, y_bres) in bresenham(gridMap, x1, y1, x2, y2):
                    valid_flag = gridMap.is_valid(x = x_bres, y = y_bres)
                    if(valid_flag):
                        gridMap.update(x = x_bres, y = y_bres, p = P_free)

                # mark laser hit spot as ocuppied (if exists)
                if dist < 20:
                    valid_flag = gridMap.is_valid(x = x2, y = y2)
                    if(valid_flag):
                        gridMap.update(x = x2, y = y2, p = P_occ)

                # for BGR image of the grid map
                X2.append(x2)
                Y2.append(y2)

            # converting grip map to BGR image
            bgr_image = gridMap.to_BGR_image()

            # marking robot position with blue pixel value
            set_pixel_color(bgr_image, x1, y1, 'BLUE')
            
            # marking neighbouring pixels with blue pixel value 
            for (x, y) in gridMap.find_neighbours(x1, y1):
                set_pixel_color(bgr_image, x, y, 'BLUE')

            # marking laser hit spots with green value
            for (x, y) in zip(X2,Y2):
                set_pixel_color(bgr_image, x, y, 'GREEN')
            

            resized_image = cv2.resize(src = bgr_image, 
                           dsize = (500, 500), 
                           interpolation = cv2.INTER_AREA)

            rotated_image = cv2.rotate(src = resized_image, 
                           rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imshow("Grid map", rotated_image)
            img_name = "./output/b" + str(i)+ "_t" + str(t)+ ".jpg"
            cv2.imwrite(img_name, rotated_image* 255.0)
            cv2.waitKey(100)

        '''
        # Saving Grid Map
        resized_image = cv2.resize(src = gridMap.to_BGR_image(), 
                        dsize = (500, 500), 
                        interpolation = cv2.INTER_AREA)

        rotated_image = cv2.rotate(src = resized_image, 
                        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        #flag_1 = cv2.imwrite(img = rotated_image * 255.0, 
        #                filename = MAPS_PATH + '/' + MAP_NAME + '_grid_map_TEST.png')
       
        cv2.imshow("Grid map", rotated_image)
        cv2.waitKey(1)
        # Calculating Maximum likelihood estimate of the map
        gridMap.calc_MLE()

        # Saving MLE of the Grid Map
        resized_image_MLE = cv2.resize(src = gridMap.to_BGR_image(), 
                            dsize = (500, 500), 
                            interpolation = cv2.INTER_AREA)

        rotated_image_MLE = cv2.rotate(src = resized_image_MLE, 
                            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow("Grid map", rotated_image_MLE)
        cv2.waitKey(1)
        
        #flag_2 = cv2.imwrite(img = rotated_image_MLE * 255.0, 
        #                filename = MAPS_PATH + '/' + MAP_NAME + '_grid_map_TEST_mle.png')
        

        #if flag_1 and flag_2:
        #    print('\nGrid map successfully saved!\n')
        

        #if cv2.waitKey(0) == 27:
        #    cv2.destroyAllWindows()

        '''

