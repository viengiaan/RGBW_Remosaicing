# LKI_RGBW (IEEE Access 2023)
An Gia Vien and Chul Lee

Official PyTorch Code for **"Learned kernel-based interpolation for efficient RGBW remosaicing"**

Paper link: https://doi.org/10.1109/ACCESS.2023.3341041

### Introduction
RGBW remosaicing is an interpolation technique that converts RGBW images captured using RGBW color filtering arrays into Bayer images. Although recent learning-based approaches using convolutional neural networks have shown substantial performance improvements, most algorithms require high computational and memory complexities, which limit their practical applicability. In this work, we propose an efficient and effective RGBW remosaicing algorithm based on learned kernel-based interpolation. First, the proposed algorithm extracts deep feature maps from input RGBW images. Then, we develop a learned kernel-based interpolation module composed of local and non-local interpolation blocks that generates two intermediate Bayer images. Specifically, the local interpolation block learns local filters to recover a Bayer image, whereas the non-local interpolation block recovers a Bayer image by estimating the non-local filters of dynamic shapes. Finally, a reconstructed Bayer image is obtained by combining the complementary information from the intermediate Bayer images using a spatially weighted fusion block. Experimental results demonstrate that the proposed algorithm achieves comparable or even better performance than state-of-the-art algorithms while providing the lowest computational and memory complexities.

### Install Libs
Running on the terminal:
```
    $ conda env create -f environment.yml
```
### Set up
- Train data path (e.g., "DATA/RGBW_Joint_Remosaic_and_Denoise/RGBW_training_dataset_fullres/")
- Test data path (e.g., "DATA/RGBW_Joint_Remosaic_and_Denoise/RGBW_validation_dataset_fullres/")
- Weight path (e.g., "WEIGHTS/Proposed/")

- Download MIPI-RGBW data set for training and testing from:
  
### Usage
Running the train code:
```
    $ CUDA_VISIBLE_DEVICES="0" python Run_test.py
```
Running the test code:
```
    $ CUDA_VISIBLE_DEVICES="0" python Run_train.py
```
### Citation
Please cite the following paper if you feel this repository useful.
```
    @inproceedings{EDWL,
        author    = {An Gia Vien and Chul Lee}, 
        title     = {Exposure-Aware Dynamic Weighted Learning for Single-Shot HDR Imaging}, 
        booktitle = {European Conference on Computer Vision},
        year      = {2022}
    }
```
### License
See [MIT License](https://github.com/viengiaan/EDWL/blob/main/LICENSE)
