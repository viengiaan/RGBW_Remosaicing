import torch
import h5py
import mat73
import scipy.io
import torch.utils.data as data
import torchvision.transforms.functional as TF
import numpy as np

class LoadData_RGBW_dataset(data.Dataset):
    def __init__(self, input_path, gt_path):
        self.in_path = input_path
        self.gt_path = gt_path
        # self.info_path = info_path

    def __getitem__(self, index):
        input_d = mat73.loadmat(self.in_path[index])
        gt_d = mat73.loadmat(self.gt_path[index])
        # info_d = mat73.loadmat(self.info_path[index])

        input = input_d['in_patch']
        input = input.astype('float32')

        gt = gt_d['gt_patch']
        gt = gt.astype('float32')
        # gt = gt.swapaxes(0, 2).swapaxes(1, 2).astype('float32')

        # info = info_d['info_patch']
        # info = info.astype('float32')

        x = torch.from_numpy(input).unsqueeze(dim = 0)
        y = torch.from_numpy(gt).unsqueeze(dim = 0) # If exist swapaxes -> no unsqueeze(dim = 0)
        # z = torch.from_numpy(info).unsqueeze(dim = 0)

        return x, y

    def __len__(self):
        return len(self.in_path)