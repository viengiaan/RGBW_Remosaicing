import os
# from cv2 import demosaicing
import torch
import h5py
import copy
import numpy as np
# import cv2 as cv
# from NETWORK.Network_ACCV import torch2cv, cv2torch

import timeit

from TOOLS.ulti import Extract_SubImages_for_BAYER, Convert2Rad_from_LogDomain

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def Generate_FULL_BAYER_image_with_GaussianWeight(input, net, device, size_patch=256, stride = 128):
 
    # if size_patch == 128:
    # Gaussian mask 1D
    gm = matlab_style_gauss2D((1, size_patch), 48) * 119.4
    gm = np.round(gm, 2)

    # Vertical split
    gm_ver = np.ones((1, size_patch))
    gm_ver[:, 0:stride] = gm[:, 0:stride]
    gm_ver = np.tile(gm_ver, (size_patch, 1))
    gm_ver = np.expand_dims(gm_ver, axis=2)
    gm_ver_inv = np.fliplr(copy.deepcopy(gm_ver))
    gm_ver_inv[:, stride:size_patch, :] = 1 - gm_ver[:, 0:stride, :]

    gm_ver = gm_ver.swapaxes(0, 2).swapaxes(1, 2)
    gm_ver = torch.from_numpy(gm_ver).float()

    gm_ver_inv = gm_ver_inv.swapaxes(0, 2).swapaxes(1, 2)
    gm_ver_inv = torch.from_numpy(gm_ver_inv.copy()).float()

    # Horizontal Split
    gm_hor = np.ones((size_patch, 1))
    gm_hor[0:stride, :] = np.transpose(gm[:, 0:stride])
    gm_hor = np.tile(gm_hor, (1, size_patch))
    gm_hor = np.expand_dims(gm_hor, axis=2)
    gm_hor_inv = np.flipud(copy.deepcopy(gm_hor))
    gm_hor_inv[stride:size_patch, :, :] = 1 - gm_hor[0:stride, :, :]

    gm_hor = gm_hor.swapaxes(0, 2).swapaxes(1, 2)
    gm_hor = torch.from_numpy(gm_hor).float()

    gm_hor_inv = gm_hor_inv.swapaxes(0, 2).swapaxes(1, 2)
    gm_hor_inv = torch.from_numpy(gm_hor_inv.copy()).float()

    # Gaussian mask 2D
    gm_2d = matlab_style_gauss2D((size_patch, size_patch), 32) * 6435
    gm_2d = np.round(gm_2d, 2)
    gm_2d = np.expand_dims(gm_2d, axis=2)
    gm_2d_inv = 1 - gm_2d

    gm_2d = gm_2d.swapaxes(0, 2).swapaxes(1, 2)
    gm_2d = torch.from_numpy(gm_2d).float()

    gm_2d_inv = gm_2d_inv.swapaxes(0, 2).swapaxes(1, 2)
    gm_2d_inv = torch.from_numpy(gm_2d_inv.copy()).float()

    # Initliaze zeros output
    channel = 1
    height = input.shape[2]
    width = input.shape[3]
    output = torch.zeros((1, channel, height, width))

    # Reconstruction
    img_width = width; img_height = height
    w_grid = [0]; h_grid = [0]

    if img_height < img_width:
        while True:
            if h_grid[-1] + size_patch < img_height:
                h_grid.append(h_grid[-1] + stride)
            if w_grid[-1] + size_patch < img_width:
                w_grid.append(w_grid[-1] + stride)
            else:
                h_grid[-1] = img_height - size_patch
                w_grid[-1] = img_width - size_patch
                break
    else:
        while True:
            if w_grid[-1] + size_patch < img_width:
                w_grid.append(w_grid[-1] + stride)
            if h_grid[-1] + size_patch < img_height:
                h_grid.append(h_grid[-1] + stride)
            else:
                h_grid[-1] = img_height - size_patch
                w_grid[-1] = img_width - size_patch
                break

    w_grid = np.array(w_grid, dtype=np.uint16)
    h_grid = np.array(h_grid, dtype=np.uint16)

    i = 0
    j = 0
    count = 0

    time_ct = 0

    PS = torch.nn.PixelShuffle(2)
 
    while i < len(h_grid):
        while j < len(w_grid):
            h = h_grid[i]
            w = w_grid[j]

            # Patch Prediction
            patch = input[:, :, h : h + size_patch, w : w + size_patch]
            patch = patch.to(device)

            start1 = timeit.default_timer()
            pout = net(patch) # NORMAL
            # _, pout = net(patch) # _, out_dc, _, _
            # pout, _ = net(patch) # out_lc, _, _, _
            stop1 = timeit.default_timer()

            # pout = PS(pout)    
            
            time_ct = time_ct + stop1 - start1
            count = count + 1

            p_cpu = pout.cpu()

            # Stitching
            new_out_p = p_cpu
            if i == 0 and j == 0:
                output[:,:, h : h + size_patch, w : w + size_patch] = new_out_p
            elif i == 0:
                hdr_patch = new_out_p * gm_ver
                output[:, :, h : h + size_patch, w - stride : w + stride] = output[:, :, h : h + size_patch, w - stride : w + stride] * gm_ver_inv
                output[:, :, h : h + size_patch, w + stride: w + size_patch] = hdr_patch[:,:,:,stride : size_patch]
                output[:, :, h : h + size_patch, w : w + stride] = output[:, :, h : h + size_patch, w : w + stride] + hdr_patch[:,:,:,0 : stride]
            elif j == 0:
                hdr_patch = new_out_p * gm_hor
                output[:,:,h - stride : h + stride, w : w + size_patch] =  output[:,:,h - stride : h + stride, w : w + size_patch] * gm_hor_inv
                output[:,:,h + stride : h + size_patch, w : w + size_patch] = hdr_patch[:,:,stride : size_patch,:]
                output[:,:,h : h + stride, w : w + size_patch] = output[:,:,h : h + stride, w : w + size_patch] + hdr_patch[:,:,0 : stride,:]
            else:
                patch2d = new_out_p * gm_2d
                patch2d_inv = output[:,:,h : h + size_patch, w : w + size_patch] * gm_2d_inv
                output[:,:,h : h + size_patch, w : w + size_patch] = patch2d + patch2d_inv
                output[:,:,h + stride : h + size_patch, w + stride : w + size_patch] = new_out_p[:,:,stride : size_patch,stride : size_patch]

            j = j + 1
        i = i + 1
        j = 0

    print('---TOTAL: %.4f/image' %(time_ct))
    print(count)

    return output

###########################################################################
###########################################################################
###########################################################################
def Torchtensor2Array(input):
    _,channel, height, width = input.size()
    output = torch.reshape(input, (channel, height, width))

    # Convert to numpy array hei x wid x 3
    HDR = output.detach().numpy()
    HDR = HDR.swapaxes(0, 2).swapaxes(0, 1)

    return HDR