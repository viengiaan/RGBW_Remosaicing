import os
import torch
import h5py
import copy
import numpy as np
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

def Generate_HDR_image_with_GaussianWeight(input, net, device, size_patch=32, stride = 16):
    if size_patch == 32:
        # Gaussian mask 1D
        gm = matlab_style_gauss2D((1, size_patch), 8) * 19
        gm = np.round(gm, 2)

        # Vertical split
        gm_ver = np.ones((1, size_patch))
        gm_ver[:, 0:stride] = gm[:, 0:stride]
        gm_ver = np.tile(gm_ver, (size_patch, 1))
        gm_ver = np.stack((gm_ver, gm_ver, gm_ver), axis=2)
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
        gm_hor = np.stack((gm_hor, gm_hor, gm_hor), axis=2)
        gm_hor_inv = np.flipud(copy.deepcopy(gm_hor))
        gm_hor_inv[stride:size_patch, :, :] = 1 - gm_hor[0:stride, :, :]

        gm_hor = gm_hor.swapaxes(0, 2).swapaxes(1, 2)
        gm_hor = torch.from_numpy(gm_hor).float()

        gm_hor_inv = gm_hor_inv.swapaxes(0, 2).swapaxes(1, 2)
        gm_hor_inv = torch.from_numpy(gm_hor_inv.copy()).float()

        # Gaussian mask 2D
        gm_2d = matlab_style_gauss2D((size_patch, size_patch), 5) * 158
        gm_2d = np.round(gm_2d, 2)
        gm_2d = np.stack((gm_2d, gm_2d, gm_2d), axis=2)
        gm_2d_inv = 1 - gm_2d

        gm_2d = gm_2d.swapaxes(0, 2).swapaxes(1, 2)
        gm_2d = torch.from_numpy(gm_2d).float()

        gm_2d_inv = gm_2d_inv.swapaxes(0, 2).swapaxes(1, 2)
        gm_2d_inv = torch.from_numpy(gm_2d_inv.copy()).float()

    if size_patch == 64:
        # Gaussian mask 1D
        gm = matlab_style_gauss2D((1, size_patch), 16) * 38
        gm = np.round(gm, 2)

        # Vertical split
        gm_ver = np.ones((1, size_patch))
        gm_ver[:, 0:stride] = gm[:, 0:stride]
        gm_ver = np.tile(gm_ver, (size_patch, 1))
        gm_ver = np.stack((gm_ver, gm_ver, gm_ver), axis=2)
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
        gm_hor = np.stack((gm_hor, gm_hor, gm_hor), axis=2)
        gm_hor_inv = np.flipud(copy.deepcopy(gm_hor))
        gm_hor_inv[stride:size_patch, :, :] = 1 - gm_hor[0:stride, :, :]

        gm_hor = gm_hor.swapaxes(0, 2).swapaxes(1, 2)
        gm_hor = torch.from_numpy(gm_hor).float()

        gm_hor_inv = gm_hor_inv.swapaxes(0, 2).swapaxes(1, 2)
        gm_hor_inv = torch.from_numpy(gm_hor_inv.copy()).float()

        # Gaussian mask 2D
        gm_2d = matlab_style_gauss2D((size_patch, size_patch), 10) * 625
        gm_2d = np.round(gm_2d, 2)
        gm_2d = np.stack((gm_2d, gm_2d, gm_2d), axis=2)
        gm_2d_inv = 1 - gm_2d

        gm_2d = gm_2d.swapaxes(0, 2).swapaxes(1, 2)
        gm_2d = torch.from_numpy(gm_2d).float()

        gm_2d_inv = gm_2d_inv.swapaxes(0, 2).swapaxes(1, 2)
        gm_2d_inv = torch.from_numpy(gm_2d_inv.copy()).float()

    if size_patch == 128:
        # Gaussian mask 1D
        gm = matlab_style_gauss2D((1, size_patch), 16) * 40
        gm = np.round(gm, 2)

        # Vertical split
        gm_ver = np.ones((1, size_patch))
        gm_ver[:, 0:stride] = gm[:, 0:stride]
        gm_ver = np.tile(gm_ver, (size_patch, 1))
        gm_ver = np.stack((gm_ver, gm_ver, gm_ver), axis=2)
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
        gm_hor = np.stack((gm_hor, gm_hor, gm_hor), axis=2)
        gm_hor_inv = np.flipud(copy.deepcopy(gm_hor))
        gm_hor_inv[stride:size_patch, :, :] = 1 - gm_hor[0:stride, :, :]

        gm_hor = gm_hor.swapaxes(0, 2).swapaxes(1, 2)
        gm_hor = torch.from_numpy(gm_hor).float()

        gm_hor_inv = gm_hor_inv.swapaxes(0, 2).swapaxes(1, 2)
        gm_hor_inv = torch.from_numpy(gm_hor_inv.copy()).float()

        # Gaussian mask 2D
        gm_2d = matlab_style_gauss2D((size_patch, size_patch), 20) * 2500
        gm_2d = np.round(gm_2d, 2)
        gm_2d = np.stack((gm_2d, gm_2d, gm_2d), axis=2)
        gm_2d_inv = 1 - gm_2d

        gm_2d = gm_2d.swapaxes(0, 2).swapaxes(1, 2)
        gm_2d = torch.from_numpy(gm_2d).float()

        gm_2d_inv = gm_2d_inv.swapaxes(0, 2).swapaxes(1, 2)
        gm_2d_inv = torch.from_numpy(gm_2d_inv.copy()).float()

    # Initliaze zeros output
    channel = 3
    height = input.shape[2]
    width = input.shape[3]
    output = torch.zeros((1, channel, height, width))

    # HDR reconstruction
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
    while i < len(h_grid):
        while j < len(w_grid):
            h = h_grid[i]
            w = w_grid[j]

            # Patch Prediction
            patch = input[:, :, h : h + size_patch, w : w + size_patch]
            patch = patch.to(device)

            Bayer_rad_log = torch.log(patch + 1e-6)
            fake_imgs = net(Bayer_rad_log)
            pout = torch.exp(fake_imgs)
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

    output[output < 0] = 0

    return output

def Generate_HDR_image_with_GaussianWeightv3(input, mask, PE_net, FS_net, Fused_net, HDR_net, device, size_patch=32, stride=16):
    if size_patch == 32:
        # Gaussian mask 1D
        gm = matlab_style_gauss2D((1, size_patch), 8) * 19
        gm = np.round(gm, 2)

        # Vertical split
        gm_ver = np.ones((1, size_patch))
        gm_ver[:, 0:stride] = gm[:, 0:stride]
        gm_ver = np.tile(gm_ver, (size_patch, 1))
        gm_ver = np.stack((gm_ver, gm_ver, gm_ver), axis=2)
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
        gm_hor = np.stack((gm_hor, gm_hor, gm_hor), axis=2)
        gm_hor_inv = np.flipud(copy.deepcopy(gm_hor))
        gm_hor_inv[stride:size_patch, :, :] = 1 - gm_hor[0:stride, :, :]

        gm_hor = gm_hor.swapaxes(0, 2).swapaxes(1, 2)
        gm_hor = torch.from_numpy(gm_hor).float()

        gm_hor_inv = gm_hor_inv.swapaxes(0, 2).swapaxes(1, 2)
        gm_hor_inv = torch.from_numpy(gm_hor_inv.copy()).float()

        # Gaussian mask 2D
        gm_2d = matlab_style_gauss2D((size_patch, size_patch), 5) * 158
        gm_2d = np.round(gm_2d, 2)
        gm_2d = np.stack((gm_2d, gm_2d, gm_2d), axis=2)
        gm_2d_inv = 1 - gm_2d

        gm_2d = gm_2d.swapaxes(0, 2).swapaxes(1, 2)
        gm_2d = torch.from_numpy(gm_2d).float()

        gm_2d_inv = gm_2d_inv.swapaxes(0, 2).swapaxes(1, 2)
        gm_2d_inv = torch.from_numpy(gm_2d_inv.copy()).float()

    if size_patch == 64:
        # Gaussian mask 1D
        gm = matlab_style_gauss2D((1, size_patch), 16) * 38
        gm = np.round(gm, 2)

        # Vertical split
        gm_ver = np.ones((1, size_patch))
        gm_ver[:, 0:stride] = gm[:, 0:stride]
        gm_ver = np.tile(gm_ver, (size_patch, 1))
        gm_ver = np.stack((gm_ver, gm_ver, gm_ver), axis=2)
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
        gm_hor = np.stack((gm_hor, gm_hor, gm_hor), axis=2)
        gm_hor_inv = np.flipud(copy.deepcopy(gm_hor))
        gm_hor_inv[stride:size_patch, :, :] = 1 - gm_hor[0:stride, :, :]

        gm_hor = gm_hor.swapaxes(0, 2).swapaxes(1, 2)
        gm_hor = torch.from_numpy(gm_hor).float()

        gm_hor_inv = gm_hor_inv.swapaxes(0, 2).swapaxes(1, 2)
        gm_hor_inv = torch.from_numpy(gm_hor_inv.copy()).float()

        # Gaussian mask 2D
        gm_2d = matlab_style_gauss2D((size_patch, size_patch), 10) * 625
        gm_2d = np.round(gm_2d, 2)
        gm_2d = np.stack((gm_2d, gm_2d, gm_2d), axis=2)
        gm_2d_inv = 1 - gm_2d

        gm_2d = gm_2d.swapaxes(0, 2).swapaxes(1, 2)
        gm_2d = torch.from_numpy(gm_2d).float()

        gm_2d_inv = gm_2d_inv.swapaxes(0, 2).swapaxes(1, 2)
        gm_2d_inv = torch.from_numpy(gm_2d_inv.copy()).float()

    if size_patch == 128:
        # Gaussian mask 1D
        gm = matlab_style_gauss2D((1, size_patch), 30) * 72
        gm = np.round(gm, 2)

        # Vertical split
        gm_ver = np.ones((1, size_patch))
        gm_ver[:, 0:stride] = gm[:, 0:stride]
        gm_ver = np.tile(gm_ver, (size_patch, 1))
        gm_ver = np.stack((gm_ver, gm_ver, gm_ver), axis=2)
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
        gm_hor = np.stack((gm_hor, gm_hor, gm_hor), axis=2)
        gm_hor_inv = np.flipud(copy.deepcopy(gm_hor))
        gm_hor_inv[stride:size_patch, :, :] = 1 - gm_hor[0:stride, :, :]

        gm_hor = gm_hor.swapaxes(0, 2).swapaxes(1, 2)
        gm_hor = torch.from_numpy(gm_hor).float()

        gm_hor_inv = gm_hor_inv.swapaxes(0, 2).swapaxes(1, 2)
        gm_hor_inv = torch.from_numpy(gm_hor_inv.copy()).float()

        # Gaussian mask 2D
        gm_2d = matlab_style_gauss2D((size_patch, size_patch), 35) * 6700
        gm_2d = np.round(gm_2d, 2)
        gm_2d = np.stack((gm_2d, gm_2d, gm_2d), axis=2)
        gm_2d_inv = 1 - gm_2d

        gm_2d = gm_2d.swapaxes(0, 2).swapaxes(1, 2)
        gm_2d = torch.from_numpy(gm_2d).float()

        gm_2d_inv = gm_2d_inv.swapaxes(0, 2).swapaxes(1, 2)
        gm_2d_inv = torch.from_numpy(gm_2d_inv.copy()).float()

    # Initliaze zeros output
    channel = 3
    height = input.shape[2]
    width = input.shape[3]
    output = torch.zeros((1, channel, height, width))

    # HDR reconstruction
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

    PixelShuffle = torch.nn.PixelShuffle(2)

    while i < len(h_grid):
        while j < len(w_grid):
            h = h_grid[i]
            w = w_grid[j]

            # Patch Prediction
            patch = input[:, :, h : h + size_patch, w : w + size_patch]
            patch = patch.to(device)

            mask_p = mask[:, :, h: h + size_patch, w: w + size_patch].to(device)

            mask_p_sub = Extract_SubImages_for_BAYER(mask_p)
            patch_sub = Extract_SubImages_for_BAYER(patch)

            x1_res = FS_net(patch)
            # x2_res = PE_net(patch)
            x2_res = PE_net(patch, mask_p)

            x1 = x1_res * mask_p_sub + patch_sub * (1. - mask_p_sub)
            x1 = PixelShuffle(x1)

            x2 = x2_res * mask_p + patch * (1. - mask_p)

            # x3 = Fused_net(x1, x2) ### GLOBAL
            x3 = Fused_net(x2, x1) ### GLOBAL & SPATIALS
            Bayer_imgs_rad = Convert2Rad_from_LogDomain(x3)

            Bayer_rad_log = torch.log(Bayer_imgs_rad + 1e-6)
            fake_imgs = HDR_net(Bayer_rad_log)
            pout = torch.exp(fake_imgs)

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

    output[output < 0] = 0

    return output

def SubBayer2Fullres(x):
    B, C, H, W = x.size()
    out = torch.zeros(B, 1, H * 2, W * 2)
    out[:, 0, 0: H * 2 : 2, 0: W * 2 : 2] = x[:, 0, 0 : H, 0 : W]
    out[:, 0, 0: H * 2 : 2, 1: W * 2 : 2] = x[:, 1, 0 : H, 0 : W]
    out[:, 0, 1: H * 2 : 2, 0: W * 2 : 2] = x[:, 2, 0 : H, 0 : W]
    out[:, 0, 1: H * 2 : 2, 1: W * 2 : 2] = x[:, 3, 0 : H, 0 : W]

    return out

def Extract_Bayer_imgs_v3(x):
    B, C, H, W = x.size()
    out_CFA = torch.zeros((B, 4, H // 2, W // 2))
    out_CFA[:, 0, 0: H // 2, 0: W // 2] = x[:, 0, 0: H: 2, 0: W: 2]
    out_CFA[:, 1, 0: H // 2, 0: W // 2] = x[:, 0, 0: H: 2, 1: W: 2]
    out_CFA[:, 2, 0: H // 2, 0: W // 2] = x[:, 0, 1: H: 2, 0: W: 2]
    out_CFA[:, 3, 0: H // 2, 0: W // 2] = x[:, 0, 1: H: 2, 1: W: 2]

    return out_CFA


def Extract_Mask_imgs(M):
    B, C, H, W = M.size()
    sub_mask = torch.zeros((B, 4, H // 2, W // 2))
    sub_mask[:, 0, 0: H // 2, 0: W // 2] = M[:, 0, 0: H: 2, 0: W: 2]
    sub_mask[:, 1, 0: H // 2, 0: W // 2] = M[:, 0, 0: H: 2, 1: W: 2]
    sub_mask[:, 2, 0: H // 2, 0: W // 2] = M[:, 0, 1: H: 2, 0: W: 2]
    sub_mask[:, 3, 0: H // 2, 0: W // 2] = M[:, 0, 1: H: 2, 1: W: 2]

    return sub_mask

def Generate_BAYER_image_PESTAGE_64x64(input, mask, PE_net, FS_net, Fused_net, device, size_patch=64, stride = 32):
    # Gaussian mask 1D
    gm = matlab_style_gauss2D((1, size_patch), 16) * 38
    gm = np.round(gm, 2)

    # Vertical split
    gm_ver = np.ones((1, size_patch))
    gm_ver[:, 0:stride] = gm[:, 0:stride]
    gm_ver = np.tile(gm_ver, (size_patch, 1))
    gm_ver = np.expand_dims(gm_ver, axis = 2)
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
    gm_2d = matlab_style_gauss2D((size_patch, size_patch), 10) * 625
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

    # HDR reconstruction
    img_width = width; img_height = height
    w_grid = [0]; h_grid = [0]

    PixelShuffle = torch.nn.PixelShuffle(2)

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
    while i < len(h_grid):
        while j < len(w_grid):
            h = h_grid[i]
            w = w_grid[j]

            # Patch Prediction
            patch = input[:, :, h : h + size_patch, w : w + size_patch]
            patch = patch.to(device)

            mask_p = mask[:, :, h : h + size_patch, w : w + size_patch]
            mask_p = mask_p.to(device)

            # Patch Prediction
            patch = input[:, :, h : h + size_patch, w : w + size_patch]
            patch = patch.to(device)

            mask_p = mask[:, :, h: h + size_patch, w: w + size_patch].to(device)

            mask_p_sub = Extract_SubImages_for_BAYER(mask_p)
            patch_sub = Extract_SubImages_for_BAYER(patch)

            x1_res = FS_net(patch)
            # x2_res = PE_net(patch)
            x2_res = PE_net(patch, mask_p)

            x1 = x1_res * mask_p_sub + patch_sub * (1. - mask_p_sub)
            x1 = PixelShuffle(x1)

            x2 = x2_res * mask_p + patch * (1. - mask_p)

            # x3 = Fused_net(x1, x2) ### GLOBAL
            x3 = Fused_net(x2, x1) ### GLOBAL & SPATIALS
            Bayer_imgs_rad = Convert2Rad_from_LogDomain(x3)

            p_cpu = Bayer_imgs_rad.cpu()

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

    output[output < 0] = 0
    # output[output > 1] = 1

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