## Proposal, Tracking and Segmentation (PTS): A Cascaded Network for Video Object Segmentation

**标签：「语义分割」「视频对象分割」 「目标跟踪」**

![image](https://github.com/Sophia-11/Awesome-CV-Paper-Scanning/blob/master/Semantic%20Segmentation/2019-07-07/images/1.png)

### 摘要
Video object segmentation (VOS) aims at pixel-level object tracking given only the annotations in the first frame. Due to the large visual variations of objects in video and the lack of training samples, it remains a difficult task despite the upsurging development of deep learning. Toward solving the VOS problem, we bring in several new insights by the proposed unified framework consisting of object proposal, tracking and segmentation components. The object proposal network transfers objectness information as generic knowledge into VOS; the tracking network identifies the target object from the proposals; and the segmentation network is performed based on the tracking results with a novel dynamic-reference based model adaptation scheme. Extensive experiments have been conducted on the DAVIS'17 dataset and the YouTube-VOS dataset, our method achieves the state-of-the-art performance on several video object segmentation benchmarks. 

视频对象分割(VOS)主要针对像素级的目标跟踪，只给出第一帧中的注释。由于视频中物体的视觉变化很大，缺乏训练样本，尽管深度学习的发展迅速，但它仍然是一项艰巨的任务。为了解决VOS问题，我们提出了由目标提案、跟踪和分割组件组成的统一框架，给我们带来了一些新的见解。对象建议网络将对象信息作为一般知识传递到VOS中；跟踪网络从建议中识别目标对象；基于跟踪结果的分割网络利用一种新的基于动态参考的模型自适应方案进行分割。在Davis的17数据集和Youtube-Vos数据集上进行了广泛的实验，我们的方法在几个视频对象分割基准上达到了最先进的性能。

### 论文网络结构
![image](https://github.com/Sophia-11/Awesome-CV-Paper-Scanning/blob/master/Semantic%20Segmentation/2019-07-07/images/2.png)

### 论文结果
![image](https://github.com/Sophia-11/Awesome-CV-Paper-Scanning/blob/master/Semantic%20Segmentation/2019-07-07/images/3.png)

### 论文地址
https://arxiv.org/abs/1907.01203v2

### code
https://github.com/sydney0zq/PTSNet
