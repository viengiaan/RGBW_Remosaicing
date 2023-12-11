from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils

from math import sqrt
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat

import numpy as np

import scipy.io

from torch.autograd import Variable

# from TOOLS.ulti import Extract_SubImages_for_BAYER

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k = 3, s = 1, p = 1):
        super().__init__()

        self.Conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p),
            nn.GELU()
        )

    def forward(self, x):

        return self.Conv2d(x)
    
#################################################################################
#################################################################################
################################ 
class Baseline_Encoder(nn.Module):
    def __init__(self, n_channels = 1, filters = [32,64,128,256]):
        super(Baseline_Encoder, self).__init__()
        self.n_channels = n_channels

        self.inc = nn.Sequential(
            Conv2d(n_channels, filters[0]),
            Conv2d(filters[0], filters[0])
        )

        self.down1 = nn.Sequential(
            Conv2d(filters[0], filters[0], k = 2, s = 2, p = 0),
            Conv2d(filters[0], filters[1]),
            Conv2d(filters[1], filters[1])
        )

        self.down2 = nn.Sequential(
            Conv2d(filters[1], filters[1], k = 2, s = 2, p = 0),
            Conv2d(filters[1], filters[2]),
            Conv2d(filters[2], filters[2])
        )
        
        self.down3 = nn.Sequential(
            Conv2d(filters[2], filters[2], k = 2, s = 2, p = 0),
            Conv2d(filters[2], filters[3]),
            Conv2d(filters[3], filters[3])
        )        


    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        return [x1, x2, x3, x4]

class Baseline_Decoder_wo_Outs(nn.Module):
    def __init__(self, filters = [32,64,128,256]):
        super(Baseline_Decoder_wo_Outs, self).__init__()

        self.up1 = nn.Sequential(
            Conv2d(filters[3], filters[3] * 2, k = 1, p = 0),
            nn.PixelShuffle(2)
        )
        self.conv1 = nn.Sequential(
            Conv2d(filters[2] * 2, filters[2]),
            Conv2d(filters[2], filters[2])
        )

        self.up2 = nn.Sequential(
            Conv2d(filters[2], filters[2] * 2, k = 1, p = 0),
            nn.PixelShuffle(2)
        )
        self.conv2 = nn.Sequential(
            Conv2d(filters[1] * 2, filters[1]),
            Conv2d(filters[1], filters[1])
        )

        self.up3 = nn.Sequential(
            Conv2d(filters[1], filters[1] * 2, k = 1, p = 0),
            nn.PixelShuffle(2)
        )
        self.conv3 = nn.Sequential(
            Conv2d(filters[0] * 2, filters[0]),
            Conv2d(filters[0], filters[0])
        )

    def forward(self, x, is_mul_outs = False):
        # [x1, x2, x3, x4] = [128, 64, 32, 16]
        out1 = self.up1(x[3])
        out1 = self.conv1(torch.cat([out1, x[2]], dim = 1))

        out2 = self.up2(out1)
        out2 = self.conv2(torch.cat([out2, x[1]], dim = 1))        
        
        out3 = self.up3(out2)
        out3 = self.conv3(torch.cat([out3, x[0]], dim = 1)) 

        if is_mul_outs:
            return out3, out2, out1

        return out3

class Baseline_Decoder(nn.Module):
    def __init__(self, out_channels = 1, filters = [32,64,128,256]):
        super(Baseline_Decoder, self).__init__()
        self.out_channels = out_channels

        self.up1 = nn.Sequential(
            Conv2d(filters[3], filters[3] * 2, k = 1, p = 0),
            nn.PixelShuffle(2)
        )
        self.conv1 = nn.Sequential(
            Conv2d(filters[2] * 2, filters[2]),
            Conv2d(filters[2], filters[2])
        )

        self.up2 = nn.Sequential(
            Conv2d(filters[2], filters[2] * 2, k = 1, p = 0),
            nn.PixelShuffle(2)
        )
        self.conv2 = nn.Sequential(
            Conv2d(filters[1] * 2, filters[1]),
            Conv2d(filters[1], filters[1])
        )

        self.up3 = nn.Sequential(
            Conv2d(filters[1], filters[1] * 2, k = 1, p = 0),
            nn.PixelShuffle(2)
        )
        self.conv3 = nn.Sequential(
            Conv2d(filters[0] * 2, filters[0]),
            Conv2d(filters[0], filters[0])
        )

        self.out_c = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # [x1, x2, x3, x4] = [128, 64, 32, 16]
        out1 = self.up1(x[3])
        out1 = self.conv1(torch.cat([out1, x[2]], dim = 1))

        out2 = self.up2(out1)
        out2 = self.conv2(torch.cat([out2, x[1]], dim = 1))        
        
        out3 = self.up3(out2)
        out3 = self.conv3(torch.cat([out3, x[0]], dim = 1))  

        out = self.out_c(out3)

        return out

##########################################################################
###########################################################################
###########################################################################
class WeightNet_Fusion(nn.Module):
    def __init__(self, in_channels = 10, out_channels = 10, kernel = 1, filters = 32):
        super(WeightNet_Fusion, self).__init__()

        # filters = 32

        self.kernel = kernel
        self.out_channels = out_channels

        self.init = Conv2d(in_channels, filters)
        
        self.conv01 = Conv2d(filters, filters)

        self.conv02 = Conv2d(filters, filters)

        self.conv03 = Conv2d(filters, filters)
        
        self.conv04 = Conv2d(filters * 2, filters)

        self.conv05 = Conv2d(filters * 2, filters)

        self.out_k = nn.Conv2d(filters * 2, out_channels * kernel * kernel, kernel_size = 3, stride = 1, padding = 1)

        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        init = self.init(x)

        conv01 = self.conv01(init)
        conv02 = self.conv02(conv01)
        conv03 = self.conv03(conv02)
        conv04 = self.conv04(torch.cat((conv02, conv03), dim = 1))
        conv05 = self.conv05(torch.cat((conv01, conv04), dim = 1))
        out_k = self.out_k(torch.cat((init, conv05), dim = 1))

        out_k = self.soft_max(out_k)

        return out_k

class DPENet_NOADD(nn.Module):
    def __init__(self, n_channels = 10, n_classes = 1, channels = 32, add = 0):
        super(DPENet_NOADD, self).__init__()
        
        self.n_channes = n_channels

        # Deformable Params (Offset + Modulation Mask + Kernel)
        self.OffsetNet = nn.Sequential(
            Conv2d(channels + add, channels),
            Conv2d(channels, channels),
            Conv2d(channels, channels),
            nn.Conv2d(channels, 2 * 3 * 3 * n_channels, kernel_size=3, padding=1, stride=1) # 2: x and y directs
        )

        self.MMNet = nn.Sequential(
            Conv2d(channels + add, channels),
            Conv2d(channels, channels),
            Conv2d(channels, channels),
            nn.Conv2d(channels, 3 * 3 * n_channels, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()           
        )

        self.KNet = nn.Sequential(
            Conv2d(channels + add, channels),
            Conv2d(channels, channels),
            Conv2d(channels, channels),
            nn.Conv2d(channels, 3 * 3 * n_channels, kernel_size=3, padding=1, stride=1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.bias_1 = nn.Conv2d(1, n_classes, kernel_size=1)
        self.bias_2 = nn.Conv2d(1, n_classes, kernel_size=1)
        self.bias_3 = nn.Conv2d(1, n_classes, kernel_size=1)
        self.bias_4 = nn.Conv2d(1, n_classes, kernel_size=1)
        self.bias_5 = nn.Conv2d(1, n_classes, kernel_size=1)
        self.bias_6 = nn.Conv2d(1, n_classes, kernel_size=1)
        self.bias_7 = nn.Conv2d(1, n_classes, kernel_size=1)
        self.bias_8 = nn.Conv2d(1, n_classes, kernel_size=1)
        self.bias_9 = nn.Conv2d(1, n_classes, kernel_size=1)
        self.bias_10 = nn.Conv2d(1, n_classes, kernel_size=1)

        self.bias = []
        self.bias.append(self.bias_1)
        self.bias.append(self.bias_2)
        self.bias.append(self.bias_3)
        self.bias.append(self.bias_4)
        self.bias.append(self.bias_5)
        self.bias.append(self.bias_6)
        self.bias.append(self.bias_7)
        self.bias.append(self.bias_8)
        self.bias.append(self.bias_9)
        self.bias.append(self.bias_10)

    def forward(self, feat, inputs):
        # Split inputs into [10, H / 2, W / 2] -> [4, H / 2, W / 2]
        # Inputs -> [W1, G11, B1, R1, G12, G21, B2, R2, G22, W2]
        offsets = self.OffsetNet(feat) # B, 2 * 3 * 3 * 10, H, W
        offsets = rearrange(offsets, 'b (c d) h w -> b c d h w', d = self.n_channes)

        modulations = self.MMNet(feat) # B, 3 * 3 * 10, H, W
        modulations = rearrange(modulations, 'b (c d) h w -> b c d h w', d = self.n_channes)

        k_norm = self.KNet(feat) # B, 90, 1, 1
        k_norm = rearrange(k_norm, 'b c h w -> (c h w) b')
        k_norm = torch.mean(k_norm, dim = 1).unsqueeze(dim = 1) # 90, 1
        k_norm = rearrange(k_norm, '(c1 k1 k2) c2 -> c2 c1 k1 k2', k1 = 3, k2 =3)

        out = []

        for i in range(0, self.n_channes):

            z = torchvision.ops.deform_conv2d(input=inputs[:,i,:,:].unsqueeze(1), offset=offsets[:,:,i,:,:], weight=k_norm[:,i,:,:].unsqueeze(1), bias=self.bias[i].bias, padding=1, stride=1, mask=modulations[:,:,i,:,:])

            out.append(z)
        
        out = torch.cat(out, dim = 1)
                
        return out

###########################################################################
###########################################################################
class DPENet_withLC(nn.Module):
    def __init__(self, n_channels = 10, n_classes = 1, channes = 32):
        super(DPENet_withLC, self).__init__()

        self.n_channes = n_channels
        
        self.DFE = DPENet_NOADD(n_channels, n_classes, channes)

        self.Local = nn.Sequential(
            Conv2d(channes, channes),
            Conv2d(channes, channes),
            Conv2d(channes, channes),
            nn.Conv2d(channes, 3 * 3 * n_channels, kernel_size=3, padding=1, stride=1),
        )

        self.unfold = nn.Unfold(kernel_size=3, padding=1, stride=1)

        self.soft_max = nn.Softmax(dim=1)

    def LC(self, inputs, local_kernel):
        B, C, H, W = inputs.size()

        inputs_resphape = self.unfold(inputs) # B x C * 9 x HW
        inputs_resphape = rearrange(inputs_resphape, 'b (c d) t -> b (t c) d', c = self.n_channes) # B x (HW*10) x 1 x 9
        inputs_resphape = inputs_resphape.unsqueeze(dim = 2) # B x HW * 10 x 1 x 9

        out = torch.einsum('bhlt, bhtv -> bhlv', [inputs_resphape, local_kernel])
        out = rearrange(out, 'b (h w d) c1 c2 -> b (c1 c2 d) h w', h = H, w = W, d = self.n_channes)

        return out   

    def forward(self, feat, inputs):

        local_kernel = self.Local(feat)
        local_kernel = rearrange(local_kernel, 'b (c d) h w -> b c d h w', d = self.n_channes) # B 9 10 H W
        local_kernel = self.soft_max(local_kernel)
        local_kernel = rearrange(local_kernel, 'b c d h w -> b (h w d) c') # B HW*10 9
        local_kernel = local_kernel.unsqueeze(dim = 3)

        out_local = self.LC(inputs, local_kernel)

        out_dc = self.DFE(feat, inputs)
        
        return out_local, out_dc
    
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
class Feature_Extractor(nn.Module):
    def __init__(self, n_channels = 1):
        super(Feature_Extractor, self).__init__()
        
        self.encoder = Baseline_Encoder(n_channels)

        self.decoder = Baseline_Decoder_wo_Outs()

        # Local and Non-local filters
        self.out = DPENet_withLC(1, n_classes=1)

    def forward(self, x, is_feature_outs = False):
        # Feature extractor
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        # Local and Non-local filters
        out_lc, out_dc = self.out(decoded, x)

        if is_feature_outs:
            return out_lc, out_dc, decoded

        return out_lc, out_dc

class LKI_RGBW(nn.Module):
    def __init__(self):
        super(LKI_RGBW, self).__init__()

        self.p1 = Feature_Extractor()
        
        self.out_Fusion = WeightNet_Fusion(2, 2)

    def forward(self, x):

        out_lc, out_dc = self.p1(x)

        out_dc[out_dc > 1.0] = 1.0
        out_dc[out_dc < 0.0] = 0.0

        out = torch.cat([out_lc, out_dc], dim = 1)
        
        W = self.out_Fusion(out)
        out_Fused = out * W
        out = torch.sum(out_Fused, dim = 1).unsqueeze(dim = 1)

        return out
        # return out_lc, out_dc
