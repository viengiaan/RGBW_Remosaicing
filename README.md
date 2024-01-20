# LKI_RGBW (IEEE Access 2023)
**An Gia Vien** and Chul Lee

Official PyTorch Code for **"[Learned Kernel-based Interpolation for Efficient RGBW Remosaicing](https://doi.org/10.1109/ACCESS.2023.3341041)"**

### Overview
![](/fig/Overview.png)

### Install Libs
Running on the terminal:
```
    $ conda env create -f environment.yml
```
### Set up
- Train data path (e.g., "DATA/RGBW_Joint_Remosaic_and_Denoise/RGBW_training_dataset_fullres/")
- Test data path (e.g., "DATA/RGBW_Joint_Remosaic_and_Denoise/RGBW_validation_dataset_fullres/")
- Weight path (e.g., "WEIGHTS/Proposed/")

- Download the MIPI-RGBW data set for training and testing from: https://1drv.ms/u/s!AoDikVPcVuNPvxs5PdON5JB0lF3V?e=uZVod0
  
### Usage
Running the test code:
```
    $ CUDA_VISIBLE_DEVICES="0" python Run_test.py
```
Running the train code:
```
    $ CUDA_VISIBLE_DEVICES="0" python Run_train.py
```
### Citation
Please cite the following paper if you feel this repository is useful.
```
@ARTICLE{Access2023_AGVien,
  author={Vien, An Gia and Lee, Chul},
  journal={IEEE Access}, 
  title={Learned Kernel-Based Interpolation for Efficient RGBW Remosaicing}, 
  year={2023},
  volume={11},
  number={},
  pages={139860-139871},
  doi={10.1109/ACCESS.2023.3341041}
}
```
### License
See [MIT License](https://github.com/viengiaan/RGBW_Remosaicing/blob/main/LICENSE)

### Acknowledgement
We followed the training code of [UNet-ResT](https://github.com/Joyies/ECCVW22-RGBW-Remosaic-Challenge).
