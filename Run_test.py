import time, math, glob
import torch
import scipy.io
import numpy as np
import sys
import os
# import imageio
import imageio.v2 as imageio
import shutil

from torchvision.utils import save_image 

from flopth import flopth

from scoring_program import kld
from scoring_program.simple_ISP import demosaic_bayer 

import cv2

from scoring_program.simple_ISP import isp_util

from TOOLS.ulti_v3 import Generate_FULL_BAYER_image_with_GaussianWeight

import pyiqa

from collections import OrderedDict

#################################### ADDTIONAL FUNCTIONS
def Extract_SubImages_for_BAYER(input):
    (batch, _, h, w) = input.size()
    DEVICE = input.device
    output = torch.zeros((batch, 4, h // 2, w // 2)).to(DEVICE).float()

    # RGGB bayer
    # G1
    output[:, 0, :, :] = input[:, 0, 0 : h : 2, 0 : w : 2]
    # B
    output[:, 1, :, :] = input[:, 0, 0 : h : 2, 1 : w : 2]
    # R
    output[:, 2, :, :] = input[:, 0, 1 : h : 2, 0 : w : 2]
    # G2
    output[:, 3, :, :] = input[:, 0, 1 : h : 2, 1 : w : 2]

    return output

def ccm_corr(rgb, CCM, fwd=True):
    '''
        apply ccm color correction on linearized rgb

    :param rgb:
        [m,n,3]. numpy float32 array
    :param CCM:
        [3x3]. ccm matrix. numpy array.
    :param fwd:
        If True, output = CCM * rgb. If False, output = inv(CCM) * rgb

    :return:
        rgb: [m,n,3]. ccm corrected RGB.

    '''

    rgb = np.clip(rgb, 0, 1)

    h, w, c = rgb.shape

    assert c == 3, 'rgb need to be in shape of [h,w,3]'

    rgb = np.reshape(rgb, (h * w, c))

    if fwd:
        rgb_ccm = np.matmul(CCM, rgb.T)
    else:
        rgb_ccm = np.matmul(np.linalg.inv(CCM), rgb.T)

    rgb_ccm = np.reshape(rgb_ccm.T, (h, w, c))

    rgb_ccm = np.clip(rgb_ccm, 0, 1)

    return rgb_ccm

def simple_ISP(bayer, cfa, is_tensor = False):

    if is_tensor:
        bayer = bayer.squeeze(dim = 0)
        # bayer = bayer.squeeze(dim = 0)
        bayer = bayer.detach().cpu()[0].numpy()  # torch tensor -> numpy array

    bayer = np.clip((bayer.astype(np.float32) - 64) / (1023 - 64), 0, 1)

    # Demosaic-net
    bayer = np.clip(np.power(bayer, 1 / 2.2), 0, 1)
    pretrained_model_path = "scoring_program/simple_ISP/pretrained_models/bayer_p/model.bin"
    demosaic_net = demosaic_bayer.get_demosaic_net_model(pretrained=pretrained_model_path, device=device, cfa='bayer', state_dict=True)

    rgb = demosaic_bayer.demosaic_by_demosaic_net(bayer=bayer, cfa=cfa, demosaic_net=demosaic_net, device=device)

    return rgb

def Final_Processing(rgb, r_gain, b_gain, CCM):

    rgb = np.power(rgb, 2.2)
    rgb = np.clip(rgb, 0, 1)

    # WBC
    rgb[:,:,0] *= r_gain
    rgb[:,:,2] *= b_gain

    rgb = np.clip(rgb, 0,1)

    # color correction
    CCM = np.asarray(CCM).astype(np.float32)
    CCM = CCM.reshape((3, 3))

    rgb = ccm_corr(rgb, CCM, fwd=True)  # (already clipped)

    # Gamma correction
    rgb = np.power(rgb, 1/2.2)
    rgb_float = rgb
    rgb = (rgb * 255).astype(np.uint8)

    return rgb, rgb_float

#####################
# CHANGE YOUR FOLDER PATH
# sys.path.insert(0, '/media/vgan/00a91feb-14ee-4d63-89c0-1bb1b7e57b8a/LOCAL/Research2023_RGBW/')

##################### Previous work
from NETWORK.Proposed import LKI_RGBW

#################################################### MIPI DATASET
num_of_images = 12

image_list = sorted(glob.glob("DATARGBW_Joint_Remosaic_and_Denoise/RGBW_validation_dataset_fullres/input/0dB" + "/*.*"))

# image_list = sorted(glob.glob("/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/RGBW_Remosaic/RGBW_Joint_Remosaic_and_Denoise/RGBW_validation_dataset_fullres/input/24dB" + "/*.*"))

# image_list = sorted(glob.glob("/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/RGBW_Remosaic/RGBW_Joint_Remosaic_and_Denoise/RGBW_validation_dataset_fullres/input/42dB" + "/*.*"))

gt_list = sorted(glob.glob("DATA/RGBW_Joint_Remosaic_and_Denoise/RGBW_validation_dataset_fullres/GT_bayer/valid_bayer_full_gt" + "/*.*"))

info_list = sorted(glob.glob("DATA/RGBW_Joint_Remosaic_and_Denoise/RGBW_validation_dataset_fullres/ImgInfo/valid_RGBW_full_imgInfo" + "/*.*"))

# CUDA_DEVICE
device = "cuda"

####################################################### Pre-Trained WEIGHT PATH
net_path = 'WEIGHTS/Proposed/model_best.pt'

########################################################## SAVE PATH
save_path = 'test/test_demo/0db/'
save_path_out = 'test/test_demo/0db/out/'

# save_path = 'test/test_demo/24db/'
# save_path_out = 'test/test_demo/24db/out/'

# save_path = 'test/test_demo/42db/'
# save_path_out = 'test/test_demo/42db/out/'

# save_path_input = 'test/Input/24db/'

save_path_gt = 'test/gt/'

########################################################## Initial (Color Interpolation)
cfa = 'GBRG'
dmsc = isp_util.dmsc_method.demosaic_net

# Creat the evaluation score path
output_filename = os.path.join(save_path, 'scores.txt')

PixelShuffle = torch.nn.PixelShuffle(2)

with torch.no_grad():

    # LOAD NETWORK
    RGBWNet = LKI_RGBW() # --------------------------------------- (MAIN)
    
    state_dict = torch.load(net_path)
    RGBWNet.load_state_dict(state_dict['model_state_dict'])
    
    RGBWNet.eval()
    RGBWNet.to(device)

    # LOAD PATH
    pretrained_model_path = "scoring_program/simple_ISP/pretrained_models/bayer_p/model.bin"
    demosaic_net = demosaic_bayer.get_demosaic_net_model(pretrained=pretrained_model_path, device=device, cfa='bayer', state_dict=True)

    # Print MODEL PARAMS
    trainable_params = sum(p.numel() for p in RGBWNet.parameters() if p.requires_grad)

    print('---Trainable params: %d' %(trainable_params))

    count = 0

    run_time = 0

    # KL-divergence on the bayer
    metric = 'KLD'
    scores = []

    score_dict = OrderedDict()
    score_dict_all = OrderedDict()

    for image_name in image_list:

        print('Image: %d' %(count + 1))

        gt_name = gt_list[count]

        info_name = info_list[count]

        input = kld.read_bin_file(image_name)
        input = input / 1023
        input = torch.from_numpy(input).float().to(device)
        input = input.unsqueeze(dim = 0)
        input = input.unsqueeze(dim = 0)

        # print(input.size())


        gt = kld.read_bin_file(gt_name)
        # gt = gt / 1023
        gt_t = torch.from_numpy(gt / 1023).float()
        gt_t = gt_t.unsqueeze(dim = 0)
        gt_t = gt_t.unsqueeze(dim = 0)

        ##############################################
        # Bayer_hat = RGBWNet(input) 
        
        ############################################## 
        Bayer_hat = Generate_FULL_BAYER_image_with_GaussianWeight(input, RGBWNet, device, size_patch=256, stride=128)
        
        Bayer_hat[Bayer_hat < 0] = 0.
        Bayer_hat[Bayer_hat > 1] = 1.
        Bayer_hat = Bayer_hat * 1023
        Bayer_hat = Bayer_hat.clamp(0, 1023)

        Bayer_hat_save = Bayer_hat / 1023

        Bayer_hat_diff = torch.abs(Bayer_hat_save.cpu() - gt_t) * 10
        # print(torch.mean(torch.pow(Bayer_hat_save.cpu() - gt_t, 2)))
        # Bayer_hat_diff = torch.abs(Bayer_hat_save - gt_t) * 1

        Bayer_input_diff = torch.abs(input.cpu() - gt_t) * 5
        
        # ####### KLD Score
        Bayer_hat = Bayer_hat.squeeze(dim = 0)
        Bayer_hat = Bayer_hat.squeeze(dim = 0)
        Bayer_hat = Bayer_hat.detach().cpu().numpy()  # torch tensor -> numpy array
        # Bayer_hat = Bayer_hat.detach().numpy()  # torch tensor -> numpy array

        # Bayer_hat = Bayer_hat * 1023

        # print(Bayer_hat.shape)
        # print(gt.shape)
        score = kld.cal_kld_main(Bayer_hat, gt)
        scores.append(score)

        print('KLD:%.4f' %(score))
        
        ###################
        # Color Interpolation
        r_gain, b_gain, CCM = isp_util.read_simpleISP_imgIno(info_name)
        rgb_hat = simple_ISP(Bayer_hat, cfa, False)

        rgb_cv = rgb_hat.detach().cpu()[0].permute(1, 2, 0).numpy()  # torch tensor -> numpy array
        rgb_cv, rgb_cv_float = Final_Processing(rgb_cv, r_gain, b_gain, CCM)

        rgb_gt = simple_ISP(gt, cfa, False)
        rgb_gt_cv = rgb_gt.detach().cpu()[0].permute(1, 2, 0).numpy()  # torch tensor -> numpy array
        rgb_gt_cv, rgb_gt_cv_float = Final_Processing(rgb_gt_cv, r_gain, b_gain, CCM)

        # rgb_gt = isp_util.predict_rgb_from_bayer_tensor(gt_raw, cfa=cfa, demosaic_net=demosaic_net, device=device)

        # #### SAVE IMAGES
        image_name = gt_name.replace('.bin', '_res.png')
        image_name_out = image_name.replace('/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/RGBW_Remosaic/RGBW_Joint_Remosaic_and_Denoise/RGBW_validation_dataset_fullres/GT_bayer/valid_bayer_full_gt/', save_path_out)
        image_name_gt = image_name.replace('/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/RGBW_Remosaic/RGBW_Joint_Remosaic_and_Denoise/RGBW_validation_dataset_fullres/GT_bayer/valid_bayer_full_gt/', save_path_gt)

        # image_name_input = image_name.replace('/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/RGBW_Remosaic/RGBW_Joint_Remosaic_and_Denoise/RGBW_validation_dataset_fullres/GT_bayer/valid_bayer_full_gt/', save_path_input)

        cv2.imwrite(image_name_out, cv2.cvtColor(rgb_cv, cv2.COLOR_RGB2BGR))
        cv2.imwrite(image_name_gt, cv2.cvtColor(rgb_gt_cv, cv2.COLOR_RGB2BGR))
        
        color_diff = np.abs(rgb_cv_float - rgb_gt_cv_float)
        color_diff = (color_diff * 5 * 255).astype(np.uint8)
        cv2.imwrite(image_name_out.replace('out','out/Diff_Color'), color_diff)

        # save_image(Bayer_hat_save, image_name_out.replace('out','out/Bayer'))
        # save_image(gt_t, image_name_out.replace('out','out/Bayer_gt'))

        # save_image(Bayer_hat_diff, image_name_out.replace('out','out/Diff_Bayer'))

        # save_image(Bayer_input_diff, image_name_out.replace('out','out/Diff_Input'))

        # save_image(input, image_name_input) # Fused image
    
        count += 1
    
    # ###############################################################
    score_dict['KLD'] = np.mean(scores)
    score_dict_all['KLD'] = scores
        
    # ############################################################### set up the metris you need.
    metrics = ['psnr', 'ssim', 'lpips']
    nr_metrics = ['musiq', 'niqe']

    for metric in metrics:
        iqa_metric = pyiqa.create_metric(metric)

        scores = []
        for item in gt_list:

            image_name = item.replace('.bin', '_res.png')
            res_bayer_name = image_name.replace('/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/RGBW_Remosaic/RGBW_Joint_Remosaic_and_Denoise/RGBW_validation_dataset_fullres/GT_bayer/valid_bayer_full_gt/', save_path_out)
            ref_bayer_name = image_name.replace('/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/RGBW_Remosaic/RGBW_Joint_Remosaic_and_Denoise/RGBW_validation_dataset_fullres/GT_bayer/valid_bayer_full_gt/', save_path_gt)    

            print(res_bayer_name)
            print(ref_bayer_name)

            res_img = imageio.imread(res_bayer_name)
            ref_img = imageio.imread(ref_bayer_name)

            # assert res_img.dtype == np.uint8 and ref_img.dtype == np.uint8, 'RGB images should be of type uint8'

            res_img = res_img.astype(np.float32)
            ref_img = ref_img.astype(np.float32)

            res_img = torch.tensor(res_img).permute(
                2, 0, 1).unsqueeze_(0) / 255.
            ref_img = torch.tensor(ref_img).permute(
                2, 0, 1).unsqueeze_(0) / 255.

            if metric not in nr_metrics:
                score = iqa_metric(res_img, ref_img).item()
            else:
                score = iqa_metric(res_img).item()  # Non-Reference assessment

            print('%s:%.4f' %(metric, score))
            scores.append(score)

        score_dict[metric] = np.mean(scores)
        score_dict_all[metric] = scores

    score_dict_all['M4'] = np.multiply(np.multiply(np.asarray(score_dict_all['psnr']), np.asarray(score_dict_all['ssim'])), np.power(2, 1 - np.asarray(score_dict_all['lpips']) - np.asarray(score_dict_all['KLD'])))
    print(score_dict_all['M4'])
    score_dict['M4'] = np.mean(score_dict_all['M4'])

    # Write the result into score_path/score.txt
    with open(output_filename, 'w') as f:
        for metric in metrics:
            f.write('{}: {}\n'.format(metric.upper(), score_dict[metric]))

        f.write('{}: {}\n'.format('KLD', score_dict['KLD']))
        f.write('{}: {}\n'.format('M4', score_dict['M4']))

    # shutil.rmtree(save_path_out)
    # shutil.rmtree(save_path_out)
