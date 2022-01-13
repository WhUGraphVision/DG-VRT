# A Novel Visual Representation on Text Using Diverse Conditional GAN for Visual Recognition

Pytorch implementation for our DCGAN. And Keras implementation for our DGVRT. 

Contact: Tao Hu (hutao_es@foxmail.com), Chunxia Xiao(cxxiao@whu.edu.cn), and Chengjiang Long (chengjiang.long@jd.com) 


## Citing DGVRT
If you find MSGAN useful in your research, please consider citing:
```
@ARTICLE{9371392,  
author={Hu, Tao and Long, Chengjiang and Xiao, Chunxia},  
journal={IEEE Transactions on Image Processing},   
title={A Novel Visual Representation on Text Using Diverse Conditional GAN for Visual Recognition},   
year={2021},  
volume={30},  
number={},  
pages={3499-3512},  
doi={10.1109/TIP.2021.3061927}}
```

## Usage

### Prerequisites
- Python 3.6
- Pytorch 0.4.0 and torchvision (https://pytorch.org/)

### Dataset
- Dataset: [Oxford 102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/bicos/) 
- Dataset: [Caltech-UCSD Birds-200-2011 Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) 

### Stage 1: Generating K synthetic images using a diverse conditional GAN (DCGAN)
- Baseline: AttnGAN <br>
- Run `cd DCGAN/code`

**Training**
- Pre-train DAMSM models:
  - For bird dataset: `python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0`
  - For flower dataset: `python pretrain_DAMSM.py --cfg cfg/DAMSM/flower.yml --gpu 0`

- Train DCGAN models:
  - For bird dataset: `python main.py --cfg cfg/bird_attnDCGAN2.yml --gpu 0`
  - For flower dataset: `python main.py --cfg cfg/flower_attnDCGAN2.yml --gpu 0`

**Validation**
- Run `python main.py --cfg cfg/eval_bird_attnDCGAN2.yml --gpu 1`
- Run `python main.py --cfg cfg/eval_flower_attnDCGAN2.yml --gpu 1`

### Stage 2: Multi-Feature Fusion for Visual Recognition using DGVRT
- Run `cd DG-VRT/code`
- For bird dataset: Run `python DGVRT-bird.py`
- For flower dataset: Run `python DGVRT-flower.py`

