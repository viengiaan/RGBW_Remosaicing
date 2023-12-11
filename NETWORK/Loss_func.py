from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Function

from torch.autograd import Variable

from scipy import signal
from math import exp

def RGB2UV(E, flag = 1):
	device = E.device

	R = E[:, 0, :, :]
	G = E[:, 1, :, :]
	B = E[:, 2, :, :]

	X = 0.4887 * R + 0.3107 * G + 0.2006 * B
	Y = 0.1762 * R + 0.8130 * G + 0.0108 * B
	Z = 0.0001 * R + 0.0102 * G + 0.9898 * B

	A = X + 15 * Y + 3 * Z
	A = A.clamp(min = 0) + 1e-6

	if flag == 1: # U
		output = 4 * X / A
	else:
		output = 9 * Y / A

	output = output * 410 / 255
	output = output.to(device)

	return output

class HVS_loss(nn.Module):
	def __init__(self):
		super(HVS_loss, self).__init__()

	def forward(self, output, gt):
		loss = torch.abs(output - gt)
		loss = torch.mean(loss)
		return loss

class Chroma_Func(nn.Module):
	def __init__(self):
		super(Chroma_Func, self).__init__()

	def forward(self, output, gt):
		loss = 0.5 * (torch.abs(RGB2UV(output, 1) - RGB2UV(gt, 1)) + torch.abs(RGB2UV(output, -1) - RGB2UV(gt, -1)))
		loss = torch.mean(loss)

		return loss

def RGB2UV_v2(E, flag = 1):
	device = E.device

	R = E[:, 0, :, :]
	G = E[:, 1, :, :]
	B = E[:, 2, :, :]

	X = 0.4124 * R + 0.2127 * G + 0.0193 * B
	Y = 0.3576 * R + 0.7152 * G + 0.1192 * B
	Z = 0.1805 * R + 0.0722 * G + 0.9504 * B

	A = X + 15 * Y + 3 * Z
	A = A + 0.1

	if flag == 1: # U
		output = 4 * X / A
	else:
		output = 9 * Y / A

	output = output * 410 / 255
	output = output.to(device)

	return output

class Chroma_Funcv2(nn.Module):
	def __init__(self):
		super(Chroma_Funcv2, self).__init__()

	def forward(self, output, gt):
		loss = 0.5 * (torch.abs(RGB2UV_v2(output, 1) - RGB2UV_v2(gt, 1)) + torch.abs(RGB2UV_v2(output, -1) - RGB2UV_v2(gt, -1)))
		loss = torch.mean(loss)

		return loss

def RGB2UV_v3(E, flag = 1):
	device = E.device

	R = E[:, 0, :, :]
	G = E[:, 1, :, :]
	B = E[:, 2, :, :]

	X = 0.4124 * R + 0.2127 * G + 0.0193 * B
	Y = 0.3576 * R + 0.7152 * G + 0.1192 * B
	Z = 0.1805 * R + 0.0722 * G + 0.9504 * B

	A = X + 15 * Y + 3 * Z
	A = A + 1e-6

	if flag == 1: # U
		output = 4 * X / A
	else:
		output = 9 * Y / A

	output = output * 410 / 255
	output = output.to(device)

	return output

class Chroma_Funcv3(nn.Module):
	def __init__(self):
		super(Chroma_Funcv3, self).__init__()

	def forward(self, output, gt):
		loss = 0.5 * (torch.abs(RGB2UV_v3(output, 1) - RGB2UV_v3(gt, 1)) + torch.abs(RGB2UV_v3(output, -1) - RGB2UV_v3(gt, -1)))
		loss = torch.mean(loss)

		return loss

class Contrast_Func(nn.Module):
	def __init__(self, channel):
		super(Contrast_Func, self).__init__()

		self.w1 = create_window(7, channel, 1.0)
		self.w2 = create_window(7, channel, 2.0)
		self.w3 = create_window(7, channel, 4.0)
		self.w4 = create_window(7, channel, 8.0)

		self.channel = channel

	def forward(self, output, gt):
		device = output.device
		w1 = self.w1.to(device)
		w2 = self.w2.to(device)
		w3 = self.w3.to(device)
		w4 = self.w4.to(device)

		cs1 = _ssim(output, gt, w1, 7, self.channel)
		cs2 = _ssim(output, gt, w2, 7, self.channel)
		cs3 = _ssim(output, gt, w3, 7, self.channel)
		cs4 = _ssim(output, gt, w4, 7, self.channel)

		cs = cs1 * cs2 * cs3 * cs4
		loss = 1. - torch.mean(cs)

		return loss

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
	return gauss / gauss.sum()

def create_window(window_size, channel, sigma):
	_1D_window = gaussian(window_size, sigma).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
	return window

def _ssim(img1, img2, window, window_size, channel):

	mu1 = F.conv2d(img1, window, padding = window_size // 2, groups = channel)

	mu2 = F.conv2d(img2, window, padding = window_size // 2, groups = channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1 * mu2

	sigma1_sq = F.conv2d(img1 * img1, window, padding = window_size // 2, groups = channel) - mu1_sq

	sigma2_sq = F.conv2d(img2 * img2, window, padding = window_size // 2, groups = channel) - mu2_sq

	sigma12 = F.conv2d(img1 * img2, window, padding = window_size // 2, groups = channel) - mu1_mu2

	C2 = (0.03 * 1)**2

	cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

	return cs

class Sobel_Grads(nn.Module):
	def __init__(self, channels):
		super(Sobel_Grads, self).__init__()

		self.c = channels

		gx = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], dtype='float32')
		self.gx = Variable(torch.from_numpy(gx).expand(self.c, 1, 3, 3).contiguous(), requires_grad=False)

		gy = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], dtype='float32')
		self.gy = Variable(torch.from_numpy(gy).expand(self.c, 1, 3, 3).contiguous(), requires_grad=False)

	def forward(self, inputs):
		device = inputs.device

		self.gx = self.gx.to(device)
		self.gy = self.gy.to(device)

		Gx = F.conv2d(inputs, self.gx, padding=1, groups=self.c)
		Gy = F.conv2d(inputs, self.gy, padding=1, groups=self.c)

		magnitude = torch.sqrt(Gx * Gx + Gy * Gy + 1e-6)

		return magnitude