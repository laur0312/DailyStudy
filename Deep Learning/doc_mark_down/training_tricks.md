---
typora-root-url: ..\img_training_tricks
---

## [ACCV2020国际细粒度识别比赛季军方案解读及Tricks汇总](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247526059&idx=2&sn=510a0aab06990f02447773b163eec0d5&chksm=ec1c8b52db6b02449ace6a4c7bb19326ac63b1bf5460978a181a0b9f62d3cd4a2660ea127818&mpshare=1&scene=1&srcid=121107RNsLINXO6ZumLHg2d4&sharer_sharetime=1607649635787&sharer_shareid=5391a0c3aeea51b626750b3a54588393&key=8ef037b02dd197d38aacb5147be04017ea48fb9ab7da548bd10e7971acf1fbed817df01a7dc497ed81ef379c73c5ece4bba444b6b9dcc7918b65b014623d12761dbe36bf72877fe4958a5b6c9deede3d469bf1ab921fb3a1638a0f67c5f3b7e1ffe29bb9c6c0209267db75d75a04442455e25b1bd6bf18686593088de59d8035&ascene=1&uin=MjE5OTA4MjI0MA%3D%3D&devicetype=Windows+10+x64&version=6300002f&lang=zh_CN&exportkey=AQX9yDocOsiCzAJHhdmcE%2Bg%3D&pass_ticket=LoZx9HH8BGGP0fxqdxOrdgIPi%2BXZEBj%2BhraC6ODYwdPNdEDYiZB%2F3mDnmEvy4PXO&wx_header=0)

- **数据清洗方案**
  1. 从1万张非三通道图片中人工挑出1000张左右的噪声图片 和 7000张左右正常图片，训练二分类噪声数据识别模型。
  2. 使用1中的二分类模型预测全量50万训练数据，挑选出阈值大于0.9的噪声数据。
  3. 使用2中噪声数据迭代 1、2过程，完成噪声数据清洗。人工检查，清洗后的训练样本中噪声数据占比小于1%。
- **标签清洗**
  1. 交叉训练，将50万训练集拆成五分，每4分当训练集，一份当测试集，训练5个模型。
  2. 将训练好的模型分别对各自测试集进行预测，将测试集top5正确的数据归为正例，top5错误的数据归为反例。
  3. 收集正例数据，重新训练模型，对反例数据进行预测，反例数据中top5正确的数据拉回放入训练集。  
  4. 使用不同的优化方案、数据增强反复迭代步骤3直至稳定（没有新的正例数据产出）。
  5. 人工干预：从反例数据中拉回5%-10%左右的数据，人工check，挑选出正例数据放入训练集
  6. 重复3、4步骤。

<p align="center">
	<img width=100% src="https://mmbiz.qpic.cn/sz_mmbiz_png/gYUsOT36vfqGnBvP4MDtWBeUcbIcaBCRjQ0gGD8koGzF8FlUAYsuMbhkgRc21nfJxr0JWzLnaDhDENfFB437Cg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1"/>  
</p>

- **清除低质量类别**
  在数据集的5000个类别中，人工看了图片数量少于50的类别，剔除其中图片混乱，无法确认此类别的具体标签。

- **数据增强**

  - mixcut
  - 随机颜色抖动
  - 随机方向——镜像翻转；4方向随机旋转
  - 随机质量——resize150~190，再放大到380；随机jpeg低质量有损压缩
  - 随机缩放贴图
  - 图片随机网格打乱重组
  - 随机crop

- **数据均衡**

  - 上采样数据均衡，每类数据采样至不少于最大类别图片数量的三分之一。

  - 统计训练数据各类别概率分布，求log后初始化fc层偏置，并在训练过程中不更新fc层偏置。参考论文：Long-tail learning via logit adjustment [通过互信息思想来缓解类别不平衡问题](https://spaces.ac.cn/archives/7615)

    ​                     **From**     ![image-20201216164709811](/softmax.png)**To**     ![image-20201216164824575](/mutual information.png)

    ```python
    import numpy as np
    import keras.backend as K
    
    
    def categorical_crossentropy_with_prior(y_true, y_pred, tau=1.0):
        """带先验分布的交叉熵
        注：y_pred不用加softmax
        """
        prior = xxxxxx  # 自己定义好prior，shape为[num_classes]
        log_prior = K.constant(np.log(prior + 1e-8))
        for _ in range(K.ndim(y_pred) - 1):
            log_prior = K.expand_dims(log_prior, 0)
        y_pred = y_pred + tau * log_prior
        return K.categorical_crossentropy(y_true, y_pred, from_logits=True)
    
    
    def sparse_categorical_crossentropy_with_prior(y_true, y_pred, tau=1.0):
        """带先验分布的稀疏交叉熵
        注：y_pred不用加softmax
        """
        prior = xxxxxx  # 自己定义好prior，shape为[num_classes]
        log_prior = K.constant(np.log(prior + 1e-8))
        for _ in range(K.ndim(y_pred) - 1):
            log_prior = K.expand_dims(log_prior, 0)
        y_pred = y_pred + tau * log_prior
        return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    ```

    

- **知识蒸馏**——Knowledge  Distillation

  - 知识蒸馏，可以将一个网络的知识转移到另一个网络，两个网络可以是同构或者异构。做法是先训练一个teacher网络，然后使用这个teacher网络的输出和数据的真实标签去训练student网络。知识蒸馏，可以用来将网络从大网络转化成一个小网络，并保留接近于大网络的性能；也可以将多个网络的学到的知识转移到一个网络中，使得单个网络的性能接近emsemble的结果。



## [总结-CNN中的目标多尺度处理](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247489720&idx=3&sn=b8c749ad9261ed450135c25b4aaeea32&chksm=ec1ff541db687c5753974b5a8159b6e7e091aa085ad6d75d69281c6b29f0e5d0ba51f88579fd&scene=21#wechat_redirect)

视觉任务中处理目标多尺度主要分为两大类：

- **图像金字塔**
- **特征金字塔**：分析CNN中的多尺度问题，其实本质上还是去分析CNN的感受野，一般认为感受野越大越好，一方面，感受野大了才能关注到大目标，另一方面，小目标可以获得更丰富的上下文信息，降低误检。



## [小目标检测](https://blog.csdn.net/m_buddy/article/details/96503647)

#### 多尺度（分辨率）对图像分类和检测的影响

1. **对图像分类的影响**
   1. 当训练图片的尺度与测试图片的尺度相差较大时性能会很差，越接近性能越好。
   2. 在高分辨下预训练得到的模型，在低分辨数据下finetune得到的结果比专门设计针对小目标的模型效果好。
2. **对目标检测的影响**
   1. <font color=blue size=3>upsample确实一定程度上可以提高性能，但是并不显著</font>，这是因为upsample提高了小目标的检测效果，但会让本来正常大小或者本来就大的目标过大，性能下降。
   2. 训练网络需要使用大量样本，样本损失会导致性能下降，丢弃样本导致训练集丰富性下降，尤其是抛弃的那个尺度的样本。

#### 难点分析

1. **网络的stride特性**

   检测网络中一般使用CNN网络作为特征提取工具，在CNN网络中为了增大感受野使得CNN网络中的特征图不断缩小，面积较小的区域的信息自然就很难传递到后面的目标检测检测器中了。

2. **训练集的分布**（参考[SNIP(2018-CVPR)](https://blog.csdn.net/m_buddy/article/details/90454642)）

   在COCO数据集中大目标和小目标的大小比值是比较大的，这就为网络适应目标带来了一定的困难。

3. **网络损失函数**

   现有的检测网络中OHEM之类的训练样本选择机制，在正负样本选择的时候对小目标并不是很友好。

4. **CNN对于尺度变化的泛化能力较弱**

   CNN学习尺度不变性较难，就算网络表现出来具有一定的尺度泛化能力，也是通过大量的参数固定下来的，因而泛化能力较弱。

#### 方法总结

1. **从图像或特征的角度**

   既然使用最后一个stage的特征去做预测很难，那么可以考虑如下优化方式：

   1. 使用FPN在多个尺度上预测不同尺度的目标；
   2. 参考[SNIPER(2018-NIPS)](https://blog.csdn.net/Gentleman_Qin/article/details/84797882)，**<font color=blue size=3>区分大小目标，针对性优化</font>**；
      - 图像金字塔，**<font color=green size=3>生成固定大小的Positive Chip和Negative Chip</font>**。
   3. 放大输入图像（用超分辨率之类的有质量的方法）（训练时图像放大1.5到2倍，预测时放大4倍）或是切图多次检测；
   4. 使用**<font color=blue size=3>空洞卷积（dilated convolution）</font>**；

2. **从anchor角度**

   1. anchor的密度

      由检测所用feature map的stride决定，这个值与前景阈值密切相关，在密集的情况下可以使anchor加倍以增加对密集目标的检测能力（TextBoxes++，Pixel-Anchor）；

   2. anchor的范围

      RetinaNet中是anchor范围是32~512，这里应根据任务检测目标的范围确定，按需调整anchor范围，或目标变化范围太大如MS COCO，这时候应采用多尺度测试；

   3. anchor的形状数量

      RetinaNet每个位置预测三尺度三比例共9个形状的anchor，这样可以增加anchor的密度，但stride决定这些形状都是同样的滑窗步进，需考虑步进会不会太大，如RetinaNet框架前景阈值是0.5时，一般anchor大小是stride的4倍左右；

3. **对于使用ROI Pooling的网络**

   A Scale-Insensitive Convolutional Neural Network for Fast Vehicle Detection(2019-T-ITS) 认为小目标在Pooling之后会导致物体结构失真，于是提出了新的**<font color=blue size=3>Context-Aware RoI Pooling</font>**方法，有助于保留有用信息，下图是该方法与简单Pooling操作的对比：

   <p align="center">
   	<img width=80% height=80% src="Context-Aware RoI Pooling.png"/>  
   </p>

   <p align="center">
   	<img width=100% height=100% src="A Scale-Insensitive Convolutional Neural Network for Fast Vehicle Detection.png"/>  
   </p>

4. **增加小目标数量**

   1. 在**<font color=blue size=3>Augmentation for small object detection</font>**中提到增加图像中小目标的数量（不影响其它目标检测的情况下，复制多个小目标），提升小目标被学习到的机会；
   2. 增加小目标图像在训练数据集中的数量，保证小目标能够被有效地学习；

5. **在对小目标的IoU阈值上**

   对小目标可以不使用严苛的阈值（0.5），可以考虑针对小目标使用**<font color=blue size=3>Cascade RCNN</font>**的思想，级联优化小目标的检测。

6. **回归损失函数上**

   在YOLO中按照不同的目标大小给了不同的损失函数加权系数：(2−*w*∗*h*)∗1.5，使用这样的策略其性能提升了1个点。

7. **小目标的GT**

   增大小目标的GT，从而变相加大目标，增加检测的能力。



## [为什么学习率不宜过小？](https://spaces.ac.cn/archives/7787)

​		Google最近发布在Arxiv上的论文《Implicit Gradient Regularization》试图回答了这个问题，它指出**<font color=red size=3>有限的学习率隐式地给优化过程带来了梯度惩罚项</font>**。学习率不宜过小，较大的学习率不仅有加速收敛的好处，还有提高模型泛化能力的好处。



## [我们真的需要把训练集的损失降低到零吗？](https://spaces.ac.cn/archives/7643)

​		以 ***b*** 为阈值，低于阈值时反而希望损失函数变大。论文把这个改动称为“Flooding”。

![image-20201217112952029](/Flooding loss.png)



<p align="center">
	<img width=80% src="https://spaces.ac.cn/usr/uploads/2020/07/2776542712.png"/>  
</p>

![image-20201217113300647](/flooding 公式详解.png)

​		平均而言，Flooding对损失函数的改动，相当于在保证了损失函数足够小之后去最小化![image-20201217113403202](/flooding_1.png)，也就是推动参数往更平稳的区域走，这通常能提供提高泛化性能（更好地抵抗扰动），因此一定程度上就能解释Flooding其作用的原因了。

