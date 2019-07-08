## Multi-Instance Multi-Scale CNN for Medical Image Classification

**标签：「图像分类」「多实例学习」 **

![](<https://github.com/Sophia-11/Awesome-CV-Paper-Review/blob/master/Image%20Classification/images/1.png>)

### 摘要
Deep learning for medical image classification faces three major challenges: 1) the number of annotated medical images for training are usually small; 2) regions of interest (ROIs) are relatively small with unclear boundaries in the whole medical images, and may appear in arbitrary positions across the x,y (and also z in 3D images) dimensions. However often only labels of the whole images are annotated, and localized ROIs are unavailable; and 3) ROIs in medical images often appear in varying sizes (scales). We approach these three challenges with a Multi-Instance Multi-Scale (MIMS) CNN: 1) We propose a multi-scale convolutional layer, which extracts patterns of different receptive fields with a shared set of convolutional kernels, so that scale-invariant patterns are captured by this compact set of kernels. As this layer contains only a small number of parameters, training on small datasets becomes feasible; 2) We propose a "top-k pooling"" to aggregate the feature maps in varying scales from multiple spatial dimensions, allowing the model to be trained using weak annotations within the multiple instance learning (MIL) framework. Our method is shown to perform well on three classification tasks involving two 3D and two 2D medical image datasets.

医学图像分类的深度学习面临三大挑战：1）用于训练的注释医学图像的数量通常很少; 2）感兴趣区域（ROI）相对较小，整个医学图像中的边界不清楚，并且可能出现在x，y（以及3D图像中的z）维度上的任意位置。但是，通常只注释整个图像的标签，并且本地化的ROI不可用; 3）医学图像中的ROI通常以不同的大小（尺度）出现。 我们使用多实例多尺度（MIMS）CNN来应对这三个挑战：1）我们提出了一个多尺度卷积层，它使用共享的卷积核集合提取不同感受域的模式，从而使得尺度不变的模式由这组紧凑的内核捕获。由于该层仅包含少量参数，因此对小型数据集的培训变得可行; 2）我们提出了一个“top-k pooling”来从多个空间维度聚合不同比例的特征图，允许在多实例学习（MIL）框架内使用弱注释训练模型。我们的方法被证明可以执行以及涉及两个3D和两个2D医学图像数据集的三个分类任务。

## 模型结构
![](<https://github.com/Sophia-11/Awesome-CV-Paper-Review/blob/master/Image%20Classification/images/2.png>)

### 论文地址
https://arxiv.org/pdf/1907.02413v1.pdf
