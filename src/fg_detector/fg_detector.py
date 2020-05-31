from collections import deque

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
import cv2
from torch.utils.data import DataLoader
from tracktor.datasets.factory import Datasets
from tracktor.config import get_output_dir
from torchvision.ops.boxes import clip_boxes_to_image, nms
import pandas as pd
import numba
from numba import int16, jit, int32
import numpy as np
import time
from scipy.signal import correlate
from scipy.ndimage import convolve1d

from scipy.ndimage import convolve1d

from torch.utils.data import DataLoader

from tracktor.datasets.factory import Datasets
from tracktor.config import get_output_dir
from torchvision.ops.boxes import clip_boxes_to_image, nms


import yaml
from tqdm import tqdm
import sacred
from sacred import Experiment


ex = Experiment()

ex.add_config('experiments/cfgs/fg_det.yaml')

def offset_image(image,pos):
    (_,height,width)= image.shape
    result=np.zeros(image.shape, dtype=np.int)
    result[:,max(0,pos[0]):min(height,height+pos[0]),max(0,pos[1]):min(width,width+pos[1])]=image[:,max(0,-pos[0]):min(height,height-pos[0]),max(0,-pos[1]):min(width,width-pos[1])]
    return result

def cross_correlate(a, b):
    mid = np.size(a) // 4
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    res = [np.dot(a[i:i + 2 * mid] - mean_a, b[mid:mid * 3] - mean_b) for i in range(2 * mid)]
    return np.array(res)


class FGDetector:
    def __init__(self, fgdet_cfg, seq):
        self.frame_interval=fgdet_cfg['frame_interval']
        self.sigma=fgdet_cfg['sigma']
        self.grid=fgdet_cfg['grid']
        self.n_best=fgdet_cfg['n_best']
        self.length=fgdet_cfg['length']
        self.sub_sampled_seq = [seq[i] for i in range(len(seq)) if i % self.frame_interval == 0]
        self.data_loader = DataLoader(self.sub_sampled_seq, batch_size=1, shuffle=False)
        return

    def calc_average_image(self, positions=None):
        self.average_imaged = None
        for i, frame in enumerate(self.data_loader):
            img = frame['img']
            if self.average_imaged is None:
                self.height = img.shape[2]
                self.width = img.shape[3]
                self.average_imaged = np.zeros((3, self.height, self.width), dtype=np.int16)
            img = 255 * img
            img = img.type(torch.uint8)
            img_np = img.numpy().astype(np.int16)
            if positions is None:
                self.average_imaged += img_np[0]
            else:
                self.average_imaged += offset_image(img_np[0],np.array(positions[i]))
        self.average_imaged=self.average_imaged//len(self.data_loader)
        return self.average_imaged



    def calc_grid_points(self):
        grid_points = [[], []]
        for axis in [0,1]:
            x = np.arange(-self.sigma * 3, self.sigma * 3)
            weights = (1 / (np.sqrt(2 * np.pi) * self.sigma) * np.exp(-0.5 * ((x / self.sigma) ** 2)))

            other_axis = [1, 0][axis]
            filtered_image_over_axis = convolve1d(self.average_imaged, weights, axis=other_axis + 1)

            grad_image = np.abs(np.gradient(filtered_image_over_axis, axis=axis + 1))
            for x_grid in range(0, self.width, self.grid):
                for y_grid in range(0, self.height, self.grid):
                    subset = np.sum(grad_image[:, y_grid:y_grid + self.grid, x_grid:x_grid + self.grid], axis=0)
                    amax = np.unravel_index(np.argmax(subset), subset.shape)
                    max = subset[amax]
                    if axis == 0 and self.length // 2 <= amax[0] + y_grid <= self.height - self.length // 2:
                        grid_points[0].append([amax[0] + y_grid, amax[1] + x_grid, max])
                    elif axis == 1 and self.length // 2 <= amax[1] <= self.width - self.length // 2:
                        grid_points[1].append([amax[0] + y_grid, amax[1] + x_grid, max])
            df = pd.DataFrame(np.array(grid_points[axis]))
            df = df.sort_values(by=[2], ascending=False)
            grid_points[axis] = df.iloc[:self.n_best].to_numpy()
        return grid_points

    def calc_positions(self, grid_points):
        num_frames = len(self.data_loader)
        series = np.zeros((num_frames, self.n_best, 2, 3, self.length))
        positions_2d=[]

        for frame_id, frame in enumerate(self.data_loader):
            img = (255 * frame['img'])
            img = img.type(torch.uint8)
            img_np = img.numpy().astype(np.int16)
            img_np = img_np.reshape(3, self.height, self.width)
            cross_corr = np.zeros((num_frames, self.n_best, 2, 3, 2 * self.length // 4))

            if frame_id == 0:
                first_image = img_np
                ref_image = first_image
                ref_series = np.zeros(( 2, 3, self.length))

            for gp_idx in range(self.n_best): # loop over all grind points
                for axis in [0,1]: #loop over x and y axis
                    if axis == 0:


                        grid_point = np.asarray(grid_points[0][gp_idx], dtype=np.int)

                        ref_series[0, :, :] = ref_image[:,
                                              grid_point[0] - self.length // 2:grid_point[0] + self.length // 2,
                                              grid_point[1]].reshape(3, self.length)
                        series[frame_id, gp_idx, 0, :, :] = img_np[:,
                                                     grid_point[0] - self.length // 2:grid_point[0] + self.length // 2,
                                                     grid_point[1]].reshape(3, self.length)
                    elif axis == 1:
                        grid_point = np.asarray(grid_points[1][gp_idx], dtype=np.int)

                        ref_series[1, :, :] = ref_image[:, grid_point[0],
                                              grid_point[1] - self.length // 2:grid_point[
                                                                                   1] + self.length // 2].reshape(3,
                                                                                                                  self.length)
                        series[frame_id, gp_idx, 1, :, :] = img_np[:, grid_point[0],
                                                     grid_point[1] - self.length // 2:grid_point[
                                                                                          1] + self.length // 2].reshape(3,
                                                                                                                         self.length)
                    for c in range(3):
                        a = cross_correlate(ref_series[axis, c, :], series[frame_id, gp_idx, axis, c, :])
                        cross_corr[frame_id, gp_idx, axis, c] = a
            all_corr = np.sum(cross_corr[frame_id, :, :, :], axis=(0, 2))
            position_2d=np.argmax(all_corr,axis=1) - self.length // 4
            positions_2d.append(position_2d)
        return np.asarray(positions_2d)

@ex.automain
def main(fg_detector, _config, _log, _run):

    torch.manual_seed(fg_detector['seed'])
    torch.cuda.manual_seed(fg_detector['seed'])
    np.random.seed(fg_detector['seed'])
    sacred.commands.print_config(_run)
    output_dir = os.path.join(get_output_dir(fg_detector['module_name']), fg_detector['name'])

    sacred_config = os.path.join(output_dir, 'sacred_config.yaml')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)
    # object detection
    _log.info("Initializing foreground detector.")
    dataset = Datasets(fg_detector['dataset'])
    fg_det=FGDetector(fg_detector, dataset[3])
    fg_det.calc_average_image()
    grid_points= fg_det.calc_grid_points()
    fg_det.cal_position(grid_points)



