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

import yaml
from tqdm import tqdm
import sacred
from sacred import Experiment


ex = Experiment()

ex.add_config('experiments/cfgs/fg_det.yaml')


class FGDetector:
    def __init__(self, fgdet_cfg):
        return

@ex.automain

def main(fg_detector,  _config, _log, _run):

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
    for seq in dataset[:1]:
        data_loader = DataLoader(seq, batch_size=1, shuffle=False)
        for i, frame in enumerate(tqdm(data_loader)):

            img=(256*frame['img'])
            img=img.type(torch.uint8)
            img_np=img.numpy()
            # img_hist=np.histogramdd(sample=img_np,bins=25, range=np.arange(0,256,8), density=True)
            # # median = torch.median(img, dim=0, keepdim=False)
            # if i > 20:
            #     break

    plt.subplot(221)
    # ax1 = plt.subplot(2, 2, 1)
    # ax1.imshow()

    print ("hallo")
