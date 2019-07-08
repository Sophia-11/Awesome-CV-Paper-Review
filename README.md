# Awesome-CV-Paper-Review

计算机视觉各个方向论文速览

# Image Classification

## 2019-07-07

### 	Multi-Instance Multi-Scale CNN for Medical Image Classification

#### **标签：「图像分类」「多实例学习」**

​               ![](<https://github.com/Sophia-11/Awesome-CV-Paper-Review/blob/master/Image%20Classification/images/1.png>)

#### 	摘要

​	     Deep learning for medical image classification faces three major challenges: 1) the number of annotated medical images for training are usually small; 2) regions of interest (ROIs) are relatively small with unclear boundaries in the whole medical images, and may appear in arbitrary positions across the x,y (and also z in 3D images) dimensions. However often only labels of the whole images are annotated, and localized ROIs are unavailable; and 3) ROIs in medical images often appear in varying sizes (scales). We approach these three challenges with a Multi-Instance Multi-Scale (MIMS) CNN: 1) We propose a multi-scale convolutional layer, which extracts patterns of different receptive fields with a shared set of convolutional kernels, so that scale-invariant patterns are captured by this compact set of kernels. As this layer contains only a small number of parameters, training on small datasets becomes feasible; 2) We propose a "top-k pooling"" to aggregate the feature maps in varying scales from multiple spatial dimensions, allowing the model to be trained using weak annotations within the multiple instance learning (MIL) framework. Our method is shown to perform well on three classification tasks involving two 3D and two 2D medical image datasets.

​             医学图像分类的深度学习面临三大挑战：1）用于训练的注释医学图像的数量通常很少; 2）感兴趣区域（ROI）相对较小，整个医学图像中的边界不清楚，并且可能出现在x，y（以及3D图像中的z）维度上的任意位置。但是，通常只注释整个图像的标签，并且本地化的ROI不可用; 3）医学图像中的ROI通常以不同的大小（尺度）出现。 我们使用多实例多尺度（MIMS）CNN来应对这三个挑战：1）我们提出了一个多尺度卷积层，它使用共享的卷积核集合提取不同感受域的模式，从而使得尺度不变的模式由这组紧凑的内核捕获。由于该层仅包含少量参数，因此对小型数据集的培训变得可行; 2）我们提出了一个“top-k pooling”来从多个空间维度聚合不同比例的特征图，允许在多实例学习（MIL）框架内使用弱注释训练模型。我们的方法被证明可以执行以及涉及两个3D和两个2D医学图像数据集的三个分类任务。

#### 模型结构

![](<https://github.com/Sophia-11/Awesome-CV-Paper-Review/blob/master/Image%20Classification/images/2.png>)

#### 论文地址

https://arxiv.org/pdf/1907.02413v1.pdf

# Semantic Segmentation

## 2019-07-08

### Proposal, Tracking and Segmentation (PTS): A Cascaded Network for Video Object Segmentation

#### **标签：「语义分割」「视频对象分割」 「目标跟踪」**

#### 摘要

Video object segmentation (VOS) aims at pixel-level object tracking given only the annotations in the first frame. Due to the large visual variations of objects in video and the lack of training samples, it remains a difficult task despite the upsurging development of deep learning. Toward solving the VOS problem, we bring in several new insights by the proposed unified framework consisting of object proposal, tracking and segmentation components. The object proposal network transfers objectness information as generic knowledge into VOS; the tracking network identifies the target object from the proposals; and the segmentation network is performed based on the tracking results with a novel dynamic-reference based model adaptation scheme. Extensive experiments have been conducted on the DAVIS'17 dataset and the YouTube-VOS dataset, our method achieves the state-of-the-art performance on several video object segmentation benchmarks. 

视频对象分割(VOS)主要针对像素级的目标跟踪，只给出第一帧中的注释。由于视频中物体的视觉变化很大，缺乏训练样本，尽管深度学习的发展迅速，但它仍然是一项艰巨的任务。为了解决VOS问题，我们提出了由目标提案、跟踪和分割组件组成的统一框架，给我们带来了一些新的见解。对象建议网络将对象信息作为一般知识传递到VOS中；跟踪网络从建议中识别目标对象；基于跟踪结果的分割网络利用一种新的基于动态参考的模型自适应方案进行分割。在Davis的17数据集和Youtube-Vos数据集上进行了广泛的实验，我们的方法在几个视频对象分割基准上达到了最先进的性能。

#### 论文地址

https://arxiv.org/abs/1907.01203v2

#### code

https://github.com/sydney0zq/PTSNet
