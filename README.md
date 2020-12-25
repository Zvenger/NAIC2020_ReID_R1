# NAIC2020 ReID 说明文档

Round1 : 重识别的行人会梦见摄像头吗   **b榜 6th**

Round2 : XMU-MAC  **b榜 25th**

(R2 由于复赛数据集扩充，服务器配置跟不上，因此无法继续，25th为使用Round1 训练的模型直接测试的排名)



## 1.如何复现R1 B榜结果

**运行环境及配置**

本次竞赛使用的服务器配置

+ 4*1080Ti 
+ CUDA 10.2
+ pytorch 1.6.0 （自带半精度加速）

根据fast-reid-master/docs/INSTALL.md进行环境配置

将本项目copy至服务器，在fast-reid-master文件夹内将logs压缩文件解压

 https://pan.baidu.com/s/1JkyFZZ0TrI1rMRU_PAOyEg 提取码：4bmh



**数据集存放**

 在fast-reid-master内新建datasets文件夹,其中文件结构如下：

naic2019

+ round1
  + train
  + train_list.txt
+ round2
  + train
  + train_list.txt

naic2020_round1

+ image_A
  + gallery
  + query
+ train
  + images
  + label.txt

**复现过程**

该结果由5个模型集成而成，其中模型均在logs/NAIC_All/A中，集成的5个模型其文件夹名分别是：

+ 0_269x
+ 0_269x_augmix
+ 0_269x_rcs_augmix
+ 1_101x_rcs
+ 2_200x_rcs

复现过程为：首先对于5个模型，各自计算dist.npy文件，最后运行fastreid下的ensemble_dist.py进行集成，获得最终的R1_B.json提交文件

分别测试五个模型：需要适当更改configs/PCL下的yml文件中的OUTPUT_DIR设置来保存文件



```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S101.yml --eval-only --num-gpus 4 MODEL.WEIGHTS logs/NAIC_All/A/1_101x_rcs/1_101x_rcs/101x_rcs_model.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S200.yml --eval-only --num-gpus 4 MODEL.WEIGHTS logs/NAIC_All/A/2_200x_rcs/1_200x_rcs/200x_rcs_model.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S269.yml --eval-only --num-gpus 4 MODEL.WEIGHTS logs/NAIC_All/A/0_269x/269x_model.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S269.yml --eval-only --num-gpus 4 MODEL.WEIGHTS logs/NAIC_All/A/0_269x_augmix/269x_augmix_model.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S269.yml --eval-only --num-gpus 4 MODEL.WEIGHTS logs/NAIC_All/A/0_269x_rcs_augmix/0_269x_rcs_augmix/269x_rcs_augmix_model.pth
```

最后在fastreid 中运行 python ensemble_dist.py
需要适当更改 ensemble_dist.py 中的query_path，gallery_path与dist1_path等路径。



## 2.如何训练和测试



**A榜如何训练和测试**

A榜训练：首先更改数据集位置， fastreid/data/bulid.py 中根据提示改为 ./datasets/

其次 configs/PCL中的yml DATASETS项 根据提示改为NAIC_All并适当更改OUTPUT_DIR保存输出日志



执行

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S50.yml --num-gpus 4
```

进行训练，其中S50.yml可改为S101.yml，S200.yml或S269.yml

训练时269x的网络使用augmix则设置DO_AUGMIX: True，否则DO_AUGMIX: False

训练时默认rcj数据增强不开启，需要在fastreid/data/transforms/bulid.py中 找到rcj的注释，取消注释

训练完成后会自行测试，如果需要另外测试，可以执行：

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S101.yml --eval-only --num-gpus 4 MODEL.WEIGHTS logs/NAIC_All/A/1_101x_rcs/1_101x_rcs/101x_rcs_model.pth
```

其中 MODEL.WEIGHTS 以及--config-file位置可适当自行更改



**B榜如何测试**

​        首先更改数据集位置， fastreid/data/bulid.py 中根据提示改为 "/home/zhangzhengjie/workspace/image_B" 可根据image_B存储的位置，自行适当修改

​        其次 configs/PCL中的yml DATASETS项 根据提示改为NAIC_test并适当更改OUTPUT_DIR保存输出日志

执行

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S101.yml --eval-only --num-gpus 4 MODEL.WEIGHTS logs/NAIC_All/A/1_101x_rcs/1_101x_rcs/101x_rcs_model.pth
```

其中MODEL.WEIGHTS 以及--config-file位置可适当自行更改



## 3.项目介绍

**模型配置位置**

configs/PCL下的yml文件，具体模型配置参数可以在yml文件中查看

**模型介绍**

+ 网络结构：
      使用 resnest 作为backbone 并且加入IBN层，在BN上由于多卡的原因使用了syncBN 最后的FC分类层使用了 circleSoftmax层替换，同时Pooling 方式由传统的avg pooling改成了可学习的gempool的方式

+ Loss：
      不带labelSmooth的CrossEntropyLoss hard TripletLoss

+ 测试：
      使用了AQE和 rerank 为了加快测试速度将batchsize设置为 512

+ 优化器Adam 使用了WarmupCosineAnnealingLR

+ 训练时将图片由256 * 128拉大至384 * 192 batchsize 64

+ 数据增强方面使用了fastreid框架自带的augmix ，以及发现的trick 0.5概率三通道随机交换（记为rcs），

同时初赛最终的方案为： 上述模型的101层版本(rcs)， 200层版本(rcs)， 269x版本， 以及269层版本（rcs），和269层版本(rcs,augmix)，总共五个模型的集成版本。

**trick**

数据增强 使用了三通道0.5概率随机交换的数据增强trick
    具体代码位置 fastreid/data/transforms/transforms.py中的RandomShuffleChannel类





**放在最后应该没人看：**

总结一下这次比赛：

首先，这我头一次参加深度学习比赛，因此初赛一开始方向就错了，这也导致后续复赛数据集扩充后，算力不够，无法继续参赛。当然就算方向对了，算力也肯定是不够的（笑），复赛复现条件是2张V100跑3天，估计是4张1080Ti可以跑半个月了，而且这个复赛总时长也才一个月。



总结一下为什么一开始方向就错了，由于是头一次参赛，所以在模型深度和模型广度的权衡上，放弃了模型广度的探索，比如增加细粒度模块，类似MGN的那种操作。由于后续实在是没法提点了，而且比赛中期组里就只有我一人还在参赛了。就考虑加深网络，从101x加深到200x以及最后的269x，现在回头看一下，实在是得不偿失，269x比101x提升也就2个点，但是也几乎多花了2倍的时间。



因为数据集扩充后，4卡1080Ti跑一次269x就得一周以上，只能放弃了。如果是101x的话，会好很多，但是也要2-3天，哎。。。莫得法子。

这次比赛参加下来，感受最深的就是方向错了这点，放弃了模型广度的探索，直接考虑加深模型深度了。当然后续估计也没机会参加其他深度学习比赛了，经验总结估计也用不上了（笑）。其他的可以吐槽的实在是太多了，此处就不表了。





