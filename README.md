# AICITY2021_Track2_DMT
The 1st place solution of track2 (Vehicle Re-Identification) in the NVIDIA AI City Challenge at CVPR 2021 Workshop. 

## Introduction

Detailed information of NVIDIA AI City Challenge 2021 can be found [here](https://www.aicitychallenge.org/).

The code is modified from [AICITY2020_DMT_VehicleReID](https://github.com/heshuting555/AICITY2020_DMT_VehicleReID), [TransReID]( https://github.com/heshuting555/TransReID )  and [reid_strong baseline]( https://github.com/michuanhaohao/reid-strong-baseline ).

## Get Started

1. `cd` to folder where you want to download this repo

2. Run `git clone https://github.com/michuanhaohao/AICITY2021_Track2_DMT.git`

3. Install dependencies: `pip install requirements.txt`

   We use cuda 11.0/python 3.7/torch 1.6.0/torchvision 0.7.0 for training and testing.

4. Prepare Datasets
		Download Original dataset, [Cropped_dataset](https://drive.google.com/file/d/1bxNjs_KZ_ocnhpsZmdMsIut93z8CqgBN/view?usp=sharing), and [SPGAN_dataset](https://drive.google.com/file/d/1nPOTrK9WUEK38mwei9yAOCMlNiF1UJXV/view?usp=sharing).
```bash

├── AIC21/
│   ├── AIC21_Track2_ReID/
│   	├── image_train/
│   	├── image_test/
│   	├── image_query/
│   	├── train_label.xml
│   	├── ...
│   	├── training_part_seg/
│   	    ├── cropped_patch/
│   	├── cropped_aic_test
│   	    ├── image_test/
│   	    ├── image_query/		
│   ├── AIC21_Track2_ReID_Simulation/
│   	├── sys_image_train/
│   	├── sys_image_train_tr/
```

5. Put pre-trained models into ./pretrained/
	-  resnet101_ibn_a-59ea0ac6.pth, densenet169_ibn_a-9f32c161.pth, resnext101_ibn_a-6ace051d.pth and se_resnet101_ibn_a-fabed4e2.pth can be downloaded from [IBN-Net](https://github.com/XingangPan/IBN-Net)
	-  resnest101-22405ba7.pth can be downloaded from [ResNest](https://github.com/zhanghang1989/ResNeSt)
	-  jx_vit_base_p16_224-80ecf9dd.pth can be downloaded from [here](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)
## Trainint and Test

We utilize 1 GPU (32GB) for training. You can train and test one backbone as follow. 

```bash
# ResNext101-IBN-a
python train.py --config_file configs/stage1/resnext101a_384.yml MODEL.DEVICE_ID "('0')"
python train_stage2_v1.py --config_file configs/stage2/resnext101a_384.yml MODEL.DEVICE_ID "('0')" OUTPUT_DIR './logs/stage2/resnext101a_384/v1'
python train_stage2_v2.py --config_file configs/stage2/resnext101a_384.yml MODEL.DEVICE_ID "('0')" OUTPUT_DIR './logs/stage2/resnext101a_384/v2'

python test.py --config_file configs/stage2/1resnext101a_384.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/resnext101a_384/v1/resnext101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/resnext101a_384/v1'
python test.py --config_file configs/stage2/resnext101a_384.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/resnext101a_384/v2/resnext101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/resnext101a_384/v2'
```
You should train camera and viewpoint models before the inference stage. You also can directly use our trained results (track_cam_rk.npy and track_view_rk.npy):

```bash
python train_cam.py --config_file configs/camera_view/camera_101a.yml
python train_view.py --config_file configs/camera_view/view_101a.yml
```

You can train all eight backbones by checking ***run.sh***. Then, you can ensemble all results:

```bash
python ensemble.py
```

All trained models can be downloaded from [here](https://drive.google.com/drive/folders/1aCQmTbYQE-mq-07q86NIMLLZRc82mc5t?usp=sharing)

## Leaderboard
|TeamName|mAP|Link|
|--------|----|-------|
|**DMT(Ours)**|0.7445|[code](https://github.com/michuanhaohao/AICITY2021_Track2_DMT)|
|NewGeneration|0.7151|[code](https://github.com/Xuanmeng-Zhang/AICITY2021-Track2)|
|CyberHu|0.6550|code|

## Citation

If you find our work useful in your research, please consider citing:
```
@inproceedings{luo2021empirical,
 title={An Empirical Study of Vehicle Re-Identification on the AI City Challenge},
 author={Luo, Hao and Chen, Weihua and Xu Xianzhe and Gu Jianyang and Zhang, Yuqi and Chong Liu and Jiang Qiyi and He, Shuting and Wang, Fan and Li, Hao},
 booktitle={Proc. CVPR Workshops},
 year={2021}
}
```
