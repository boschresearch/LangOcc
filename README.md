# LangOcc: Open Vocabulary Occupancy Estimation via Volume Rendering 
Official Implementation of "LangOcc: Open Vocabulary Occupancy Estimation via Volume Rendering" by Boeder et al., presented at the 3DV 2025 conference.

[![arXiv](https://img.shields.io/badge/arXiv-2401.08815-red)](https://arxiv.org/abs/2407.17310)

## Installation

### 1. Create virtual env
```shell script
conda create -n langocc python=3.8 -y
conda activate langocc
```

### 2. Install dependencies
Please make sure to have CUDA 11.3 installed and in your PATH.

```shell script
# install pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# install openmim, used for installing mmcv
pip install -U openmim

# install mmcv
mim install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# install mmdet, mmsegmentation and ninja
pip install mmdet==2.25.1 ninja==1.11.1
```

### 3. Install LangOcc
Assuming your terminal is in the langocc directory:
```shell script
pip install -v -e .
```

### 4. Install MaskCLIP
We also need to install MaskCLIP to generate the ground truth vision-language features. Swap to the `MaskCLIP` directory and follow these steps:
```shell script
cd MaskCLIP
# Install MaskCLIP requirements
pip install -r requirements.txt
pip install --no-cache-dir opencv-python
# Install CLIP
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
# Install MaskCLIP
pip install --no-cache-dir -v -e .
# Change back to root directory of the repo
cd ..
```

## Data Preparation
1. Please create a directory `./data` in the root directory of the repository.

2. Download nuScenes AND nuScenes-panoptic [https://www.nuscenes.org/download].

3. Download the Occ3D-nuScenes dataset from [https://github.com/Tsinghua-MARS-Lab/Occ3D]. The download link can be found in their README.md.

4. Download the Open Vocabulary Benchmark from [here](https://data.ciirc.cvut.cz/public/projects/2023POP3D/retrieval_benchmark.tar.gz), extract the contents and rename the directory to `retrieval_benchmark`.

5. Download the info files of MaskCLIP for [train](https://data.ciirc.cvut.cz/public/projects/2023POP3D/nuscenes_infos_train.pkl) and [val](https://data.ciirc.cvut.cz/public/projects/2023POP3D/nuscenes_infos_val.pkl).

6. Generate the annotation files.  This will put the annotation files into the `./data` directory by default. The process can take up to ~1h.
```shell script
python tools/create_data_bevdet.py
python tools/create_data_bevdet.py --version v1.0-test # we also need the test info files for the open vocabulary benchmark
```

7. Copy or softlink the files into the `./data` directory. The structure of the data directory should be as follows:

```shell script
data
├── nuscenes
│  ├── v1.0-trainval (Step 2, nuScenes+nuScenes-panoptic files)
│  ├── sweeps (Step 2, nuScenes files)
│  ├── samples (Step 2, nuScenes files)
│  └── panoptic (Step 2, nuScenes-panoptic files)
├── gts (Step 3)
├── retrieval_benchmark (Step 4)
├── nuscenes_infos_train.pkl (Step 5)
├── nuscenes_infos_val.pkl (Step 5)
├── bevdetv2-nuscenes_infos_train.pkl (Step 6)
├── bevdetv2-nuscenes_infos_val.pkl (Step 6)
├── bevdetv2-nuscenes_infos_test.pkl (Step 6)
├── rays (See next chapter)
└── embeddings (See next chapter)
```

8. Download pretrained backbone weights for ResNet-50. Because the original download link does not work anymore, please find an anonymized link below.
Download the checkpoint, create a directory `./ckpts` and put the file in there.  
Downloadlink: 

## Generate Ground Truth Features
We recommend to create two directories `embeddings` and `rays` in a location with enough disk space and softlink them into `./data`, as the following scripts will write data to these locations (the `./data` directory should look like in the tree above).
1. First, we pre-generate all training rays, so that we do not have to store the complete feature maps (~7 GB).
```shell script
cd MaskCLIP
python tools/generate_rays.py --exact
```
2. Next, we need to prepare the MaskCLIP weights from the original CLIP model.
```shell script
# In the MaskCLIP directory:
mkdir -p ./pretrain
python tools/maskclip_utils/convert_clip_weights.py --model ViT16 --backbone
python tools/maskclip_utils/convert_clip_weights.py --model ViT16
```
3. Download the pre-trained MaskCLIP weights from [this link](https://entuedu-my.sharepoint.com/:u:/g/personal/chong033_e_ntu_edu_sg/EZSrnPaNFwNNqpECYCkgpg4BEjN782MUD7ZUEPXFWSTEXA?e=mOaseS) and put them to `ckpts/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k.pth`.

4. Afterwards, we can start generating the features for each ray. This process can take a long time, and takes ~535 GB of storage.
```shell script
# In the MaskCLIP directory:
python tools/extract_features.py configs/maskclip_plus/anno_free/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k__nuscenes_trainvaltest.py --save-dir ../data/embeddings/MaskCLIP --checkpoint ckpts/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k.pth --complete --sample
```
You can speed up the process by starting multiple instances of the generation script that each handle different token ranges. For example, if you want to parallelize 4 scripts, you could do:
```shell script
# In the MaskCLIP directory:
python tools/extract_features.py configs/maskclip_plus/anno_free/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k__nuscenes_trainvaltest.py --save-dir ../data/embeddings/MaskCLIP --checkpoint ckpts/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k.pth --complete --sample --start 0 --end 8538

python tools/extract_features.py configs/maskclip_plus/anno_free/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k__nuscenes_trainvaltest.py --save-dir ../data/embeddings/MaskCLIP --checkpoint ckpts/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k.pth --complete --sample --start 8538 --end 17074

python tools/extract_features.py configs/maskclip_plus/anno_free/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k__nuscenes_trainvaltest.py --save-dir ../data/embeddings/MaskCLIP --checkpoint ckpts/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k.pth --complete --sample --start 17074 --end 25611

python tools/extract_features.py configs/maskclip_plus/anno_free/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k__nuscenes_trainvaltest.py --save-dir ../data/embeddings/MaskCLIP --checkpoint ckpts/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k.pth --complete --sample --start 25611 --end 34150
```

## Train model
We provide configuration files for the full and reduced version we use in the paper in the `./configs` directory. 

### "Full" model
If you want to train the LangOcc (Full) model:
```shell
# In the root directory of the repository:
# single gpu
python tools/train.py configs/lang_occ/lang-occ_full.py
# multiple gpu (replace "num_gpu" with the number of available GPUs) - 4 GPU's are reccomended.
./tools/dist_train.sh configs/lang_occ/lang-occ_full.py num_gpu
```
In order to reproduce the results of the paper, please use 4 GPU's, so that the learning rate remains unchanged.
Also, due to some non-deterministic operations, the results may deviate slightly (up or down) from the results presented in the paper.

### "Reduced" Model
If you want to train the LangOcc (Reduced) model, you need to first train the reducer. 
To do that, we first precompute the CLIP features of the vocabulary:
```shell
# In the root directory
python tools/create_class_embeddings.py --use-templates
```
Then, we can train the reducer model:
```shell
# In the root directory
python tools/train_clip_reducer.py --use-templates
```
Note that the loss of this training might go to `nan`, but this is okay.

Afterwards, we can train the LangOcc (Reduced) model:
```shell
# single gpu
python tools/train.py configs/lang_occ/lang-occ_reduced.py
# multiple gpu (replace "num_gpu" with the number of available GPUs) - 4 GPU's are reccomended.
./tools/dist_train.sh configs/lang_occ/lang-occ_reduced.py num_gpu
```

## Test model
After training, you can test the model on the open vocabulary benchmark or on Occ3D-nuScenes.
1. Evaluate on the Open Vocabulary Retrieval benchmark:
```shell
# In the root directory:
python tools/eval_open_vocab.py --cfg lang-occ_full --ckpt epoch_18_ema --use-templates
```

2. Evaluate on the Occ3D-nuScenes benchmark:  
Before evaluation, we precompute the CLIP features of the vocabulary we use to assign class labels (if you have not done this already in the step above).
```shell
python tools/create_class_embeddings.py --use-templates
```

Afterwards, we can start the evaluation:
```shell
# single gpu
python tools/test.py configs/lang_occ/lang-occ_full.py work_dirs/lang-occ_full/epoch_18_ema.pth --eval mIoU --use-templates
# multiple gpu
./tools/dist_test.sh configs/lang_occ/lang-occ_full.py work_dirs/lang-occ_full/epoch_18_ema.pth num_gpu --eval mIoU --use-templates
```

You can also store the predicted occupancy for visualization by using the `--save-occ-path` flag:
```shell
# multiple gpu
./tools/dist_test.sh configs/lang_occ/lang-occ_full.py work_dirs/lang-occ_full/epoch_18_ema.pth num_gpu --eval mIoU --save-occ-path ./occ
```

## Common issues
In the following, we list some common errors you might encounter while installing or running this repository, and how to fix them:

1. **No kernel image found for bev_pool_v2**  
If you encounter this error, please uninstall mmdet3d again and make sure you have CUDA 11.3 installed and in your path. Also make sure you have
`ninja==1.11.1` installed via pip. Then run `pip install -v -e .` again to compile the kernel images again.

2. **Error: "from numba.np.ufunc import _internal SystemError: initialization of _internal failed without raising an exception"**  
In this case, please install the numpy version 1.23.5 via `pip install numpy==1.23.5`.

3. **Training stuck, no training logs printed**  
Sometimes the `nerfacc` extension will put a lock on the cuda files. If you do not see any training iteration logs after ~5mins, this might be the issue. Please interrupt the run and remove the lock under `~/.cache/torch_extensions/py38_cu113/nerfacc_cuda/lock`. Restart the training afterwards.

4. **Resume runs**
If the training is interrupted at any point and you want to resume from a checkpoint, you can simply use the `--resume-from` command as follows:
``` shell
./tools/dist_train.sh configs/lang_occ/lang-occ_full.py num_gpu --resume-from /path/to/checkpoint/latest.pth
```
The checkpoints are usually saved under the `work_dirs` directory. By default, a checkpoint is created every 4 epochs.

5. **Environment**  
Please note that this code has only been tested on Linux machines. It is not guaranteed to work on Windows.

## License
This project is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE) file for details.

For a list of other open source components included in this project, see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).

## Purpose of the project
This software is a research prototype, solely developed for and published as part of the publication cited above.

## Contact     
Please feel free to open an issue or contact personally if you have questions, need help, or need explanations. Don't hesitate to write an email to the following email address:
simon.boeder@de.bosch.com

## References
The codebase is forked from BEVDet (https://github.com/HuangJunJie2017/BEVDet).


Copyright (c) 2022 Robert Bosch GmbH  
SPDX-License-Identifier: AGPL-3.0