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

from scipy.ndimage import convolve1d

from torch.utils.data import DataLoader

from tracktor.datasets.factory import Datasets
from tracktor.config import get_output_dir


import yaml
from tqdm import tqdm
import sacred
from sacred import Experiment



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

@jit(int16[:,:,:,:](int16[:,:,:],int16[:,:,:,:]))
def add2histograms(img_np,all_histograms): # Function is compiled and runs in machine code
    height=img_np.shape[1]
    width=img_np.shape[2]
    for x in range(width):
        for y in range(height):
            for c in range(3):
                val = img_np[c,y,x]
                all_histograms[val,c,y,x]+=1
    return all_histograms

def im2show(img):
    return np.swapaxes(np.swapaxes(img, 0, 2),1,0)


@jit(int16[:,:,:,:](int16[:,:,:,:],int32))
def find_bg(all_histograms, color_scale_factor): # Function is compiled and runs in machine code
    '''
    calculates the BG of the
    '''

    height=all_histograms.shape[2]
    width=all_histograms.shape[3]
    bg=np.zeros((2,3,height,width),dtype=np.int16)
    for x in range(width):
        for y in range(height):
            for c in range(3):
                arg=np.argmax(all_histograms[:,c,y,x])
                bg[0,c,y,x]= color_scale_factor * arg
                bg[1,c,y,x]= all_histograms[arg,c,y,x]
    return bg

def frame_to_np_img(frame, color_scale_factor=1):
    img = ((256 // color_scale_factor - 1) * frame['img'])
    img = img.type(torch.uint8)
    return img.numpy().astype(np.int16)[0]



class FGDetector:
    def __init__(self, fgdet_cfg, seq):
        self.frame_interval=fgdet_cfg['frame_interval']
        self.sigma=fgdet_cfg['sigma']
        self.grid=fgdet_cfg['grid']
        self.n_best=fgdet_cfg['n_best']
        self.length=fgdet_cfg['length']
        self.color_scale_factor=fgdet_cfg['color_scale_factor']
        self.fg_delta_mu=fgdet_cfg['fg_delta_mu']
        self.fg_delta_sigma=fgdet_cfg['fg_delta_sigma']
        self.color_scale_factor=fgdet_cfg['color_scale_factor']

        self.sub_sampled_seq = [seq[i] for i in range(len(seq)) if i % self.frame_interval == 0]
        self.data_loader = DataLoader(self.sub_sampled_seq, batch_size=1, shuffle=False)
        return

    def calc_average_image(self, positions=None):

        self.average_imaged = None
        for i, frame in enumerate(self.data_loader):

            if self.average_imaged is None:
                img = frame['img']
                self.height = img.shape[2]
                self.width = img.shape[3]
                self.average_imaged = np.zeros((3, self.height, self.width), dtype=np.int16)
            img_np = frame_to_np_img(frame)

            if positions is None:
                self.average_imaged += img_np
            else:
                self.average_imaged += offset_image(img_np,np.array(positions[i]))
        self.average_imaged=self.average_imaged//len(self.data_loader)
        return self.average_imaged



    def calc_grid_points(self):
        self.grid_points = [[], []]
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
                        self.grid_points[0].append([amax[0] + y_grid, amax[1] + x_grid, max])
                    elif axis == 1 and self.length // 2 <= amax[1] <= self.width - self.length // 2:
                        self.grid_points[1].append([amax[0] + y_grid, amax[1] + x_grid, max])
            df = pd.DataFrame(np.array(self.grid_points[axis]))
            df = df.sort_values(by=[2], ascending=False)
            self.grid_points[axis] = df.iloc[:self.n_best].to_numpy()

        return self.grid_points

    def calc_positions(self):
        positions_2d=[]
        for frame_id, frame in enumerate(self.data_loader):
            if frame_id == 0:
                self.ref_image = frame_to_np_img(frame)
            position_2d=self.calc_position(frame)
            positions_2d.append(position_2d)
        self.positions=np.array(positions_2d)
        return self.positions

    def calc_position(self, frame):

        ref_series = np.zeros((2, 3, self.length))
        series = np.zeros((self.n_best, 2, 3, self.length))
        img_np = frame_to_np_img(frame)
        cross_corr = np.zeros(( self.n_best, 2, 3, 2 * self.length // 4))

        for gp_idx in range(self.n_best): # loop over all grind points
            for axis in [0,1]: #loop over x and y axis
                if axis == 0:

                    grid_point = np.asarray(self.grid_points[0][gp_idx], dtype=np.int)

                    ref_series[0, :, :] = self.ref_image[:,
                                          grid_point[0] - self.length // 2:grid_point[0] + self.length // 2,
                                          grid_point[1]].reshape(3, self.length)
                    series[ gp_idx, 0, :, :] = img_np[:,
                                                 grid_point[0] - self.length // 2:grid_point[0] + self.length // 2,
                                                 grid_point[1]].reshape(3, self.length)
                elif axis == 1:
                    grid_point = np.asarray(self.grid_points[1][gp_idx], dtype=np.int)

                    ref_series[1, :, :] = self.ref_image[:, grid_point[0],
                                          grid_point[1] - self.length // 2:grid_point[
                                                                               1] + self.length // 2].reshape(3,
                                                                                                              self.length)
                    series[ gp_idx, 1, :, :] = img_np[:, grid_point[0],
                                                 grid_point[1] - self.length // 2:grid_point[
                                                                                      1] + self.length // 2].reshape(3,
                                                                                                                     self.length)
                for c in range(3):
                    a = cross_correlate(ref_series[axis, c, :], series[ gp_idx, axis, c, :])
                    cross_corr[ gp_idx, axis, c] = a
        all_corr = np.sum(cross_corr[ :, :, :], axis=(0, 2))
        position_2d=np.argmax(all_corr,axis=1) - self.length // 4
        return position_2d


    def calc_histograms(self):

        self.all_histograms = np.zeros((256 // self.color_scale_factor, 3, self.height, self.width), dtype=np.int16)
        for i, frame in enumerate(self.data_loader):
            x= frame_to_np_img(frame, self.color_scale_factor)
            assert x.shape[1]==self.height
            assert x.shape[2]==self.width
            img_offset= offset_image(x,np.array(self.positions[i])).astype(np.int16)
            self.all_histograms=add2histograms(img_offset,self.all_histograms)

    def calc_bg(self):
        self.bg = find_bg(self.all_histograms, self.color_scale_factor)
        return self.bg


    def calc_fg(self,frame, motion_compensation=True):
        x = frame_to_np_img(frame)
        if motion_compensation:
            position=self.calc_position(frame)
        else:
            position=[0,0]
        img_offset = offset_image(x, np.array(position).astype(np.int16))
        delta_bg = np.sum((img_offset - self.bg[0])**2,axis=0)
        self.fg = 1 / (1 + np.exp(-(delta_bg - self.fg_delta_mu) / self.fg_delta_sigma**2))
        return self.fg




ex = Experiment()
ex.add_config('experiments/cfgs/fg_det.yaml')

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
    fg_det.calc_positions(grid_points)



