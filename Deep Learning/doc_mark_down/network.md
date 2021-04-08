---
typora-root-url: ..\img_network
---

## CycleGAN

<p align="center">
	<img width=60% src="Cycle-GAN.png"/>  
</p>



<p align="center">
	<img width=100% height=100% src="Cycle-GAN generator.png"/>  
</p>

**<font color=green size=3>loss</font>**

对于discriminator A：

<p align="center">
	<img width=60% height=100% src="cycle gan loss d_a.png"/>  
</p>

对于discriminator B：

<p align="center">
	<img width=60% height=100% src="cycle gan loss d_b.png"/>  
</p>

对于generator BA：

<p align="center">
	<img width=60% height=100% src="cycle gan loss g_ba.png"/>  
</p>

对于generator AB： 

<p align="center">
	<img width=60% height=100% src="cycle gan loss g_ab.png"/>  
</p>

​		对于generator添加重构误差项（cycle consistency loss），跟对偶学习一样，能够引导两个generator更好地完成encode和decode的任务。而两个D则起到纠正编码结果符合某个domain的风格的作用。



**<font color=green size=3>Tricks</font>**

- 在训练Discriminator的时候，不再仅仅把Generator最新生成的图片作为训练集喂给D，而是加上缓存的一些G在之前生成的图片， 这样可以一定程度上避免模型震荡。



**<font color=green size=3>Q & A</font>**

Q：去掉重构误差，模型是否还有效？

A：模型仍然有效，只是收敛比较慢，毕竟缺少了重构误差这样的强引导信息。以及，虽然实现了风格迁移，但是人物的一些属性改变了，比如可能出现『变性』、『变脸』，而姿态在转换的时候一般不出现错误。这表明，**对偶重构误差能够引导模型在迁移的时候保留图像固有的属性；而对抗loss则负责确定模型该学什么，该怎么迁移**。



<p align="center">
	<img width=60% height=60% src="Cycle-GAN Half Cycle.png"/>  
</p>



Q：能不能不要完整的cycle，只做一半？

A：结论是不行，缺少了对偶的部分，就少了重构误差，仅仅依靠D_B来纠正G_AB是不够的。从讨论1来看，对偶的作用还是很大的，即使缺少了重构误差。



## HRNet

- 网络结构设计思路

<p align="center">
	<img width=100% height=100% src="Neural architecture design.png"/>  
</p>

- 不同于分类任务，人体姿态识别这类的任务中，需要生成一个高分辨率的heatmap来进行关键点检测。

<p align="center">
	<img width=80% height=80% src="high-resolution networks vs classification networks.jpg"/>  
</p>

- HRNet

  - 获取高分辨率的一般方式：先降分辨率，然后再升分辨率

    <p align="center">
    	<img width=100% height=100% src="Recovering high-resolution from low-resolution.png"/>  
    </p>

  - HRNet结构

    <p align="center">
    	<img width=80% height=80% src="HRNet.webp"/>  
    </p>

    - fusion

      <p align="center">
      	<img width=100% height=100% src="Fusion in HRNet.webp"/>  
      </p>

      - 同分辨率的层直接复制。
      - 需要升分辨率的使用bilinear upsample + 1x1卷积将channel数统一。
      - 需要降分辨率的使用strided 3x3 卷积。
        - 至于为何要用strided 3x3卷积，这是因为卷积在降维的时候会出现信息损失，**<font color=green size=3>使用strided 3x3卷积是为了通过学习的方式，降低信息的损耗</font>**。所以这里没有用maxpool或者组合池化。
      - 三个feature map融合的方式是相加。

    - 输出特征融合

      <p align="center">
      	<img width=100% height=100% src="Merge methods for HRNet.webp"/>  
      </p>

      <p align="center">
      	<img width=100% height=100% src="Merge 4 HRNet.jpg"/>  
      </p>

      - (a)图展示的是HRNetV1的特征选择，只使用分辨率最高的特征图。
      - (b)图展示的是HRNetV2的特征选择，将所有分辨率的特征图（小的特征图进行upsample）进行concate，**<font color=blue size=3>主要用于语义分割和面部关键点检测。</font>**
      - (c)图展示的是HRNetV2p的特征选择，在HRNetV2的基础上，**<font color=blue size=3>使用strided 3x3卷积是为了通过学习的方式，降低信息的损耗。</font>**
      - (d)图展示的是HRNetV2，采用上图的融合方式，**<font color=blue size=3>主要用于训练分类网络。</font>**



## Residual Network

#### residual block

<p align="center">
	<img width=100% src="residual block.png"/>  
</p>

​		**残差的引入去掉了主体部分，从而突出了微小的变化**。

#### bottle-neck block

<p align="center">
	<img width=100% src="two types of residual block.png"/>  
</p>

​		在训练浅层网络的时候，我们选用前面这种，而如果网络较深(大于50层)时，会考虑使用后面这种(bottleneck)，这两个设计具有相似的时间复杂度。

#### residual unit

<p align="center">
	<img width=100% src="full pre-activation in residul block.png"/>  
</p>

<p align="center">
	<img width=100% src="full pre-activation in residul block performance.png"/>  
</p>

- **(a) original**：原始的结构
- **(b) BN after addition**：这是在做相反的实验，本来我们的目的是把ReLU移到旁路上去，这里反而把BN拿出来，这进一步破坏了主路线上的恒等关系，阻碍了信号的传递，从结果也很容易看出，这种做法不ok
- **(c) ReLU before addition**：将 *f* 变为恒等变换，最容易想到的方法就是将ReLU直接移动到BN后面，但这会出现一个问题，一个残差函数的输出应该可以是 实数空间，但是经过ReLU之后就会变为正实数，这种做法的结果也比 (a) 要差。

直接提上来似乎不行，但是问题反过来想， 在addition之后做ReLU，不是相当于在下一次conv之前做ReLU吗？

- **(d) ReLU-only pre-activation：**根据刚才的想法，我们把ReLU放到前面去，然而我们得到的结果和 (a) 差不多，原因是什么呢？因为这个ReLU层不与BN层连接使用，因此无法共享BN所带来的好处。
- **(e) full pre-activation：**那要不我们也把BN弄前面去，惊喜出现了，我们得到了相当可观的结果，是的，这便是我们最后要使用的Unit结构！！！



## Cascade RCNN

1. mismatch

   <p align="center">
   	<img width=50% height=50% src="Mismatch in RCNN.jpg"/>
   </p>

   training阶段和inference阶段，第二阶段bbox回归器的输入分布是不一样的：training阶段的输入proposals质量更高（被采样过，IoU>threshold），inference阶段的输入proposals质量相对较差（没有被采样过，可能包括很多IoU<threshold的），这就是论文中提到**mismatch**问题，这个问题是固有存在的，通常threshold取0.5时，mismatch问题还不会很严重。

2. Cascade RCNN

   <p align="center">
   	<img width=100% height=100% src="Architecture of different frameworks in Cascade RCNN.jpg"/>  
   </p>

   - RPN提出的proposals大部分质量不高，导致没办法直接使用高阈值的detector，Cascade R-CNN使用cascade回归作为一种重采样的机制，逐stage提高proposal的IoU值，从而使得前一个stage重新采样过的proposals能够适应下一个有更高阈值的stage。
     - 每一个stage的detector都不会过拟合，都有足够满足阈值条件的样本。
     - 更深层的detector也就可以优化更大阈值的proposals。
     - 每个stage的H不相同，意味着可以适应多级的分布。
     - 在inference时，虽然最开始RPN提出的proposals质量依然不高，但在每经过一个stage后质量都会提高，从而和有更高IoU阈值的detector之间不会有很严重的mismatch。



## Siamese Network

<p align="center">
	<img width=80% src="siamese network.png"/>  
</p>

<p align="center">
	<img width=100% src="siamese network.jpg"/>  
</p>

​		孪生神经网络（siamese network）中，其采用的损失函数是contrastive loss，这种损失函数可以有效的处理孪生神经网络中的paired data的关系。

<p align="center">
	<img width=80% src="contrastive loss.png"/>  
</p>

<p align="center">
	<img width=80% src="contrastive loss_1.png"/>  
</p>

代表两个样本特征的欧氏距离（二范数），P 表示样本的特征维数，Y 为两个样本是否匹配的标签，Y=1 代表两个样本相似或者匹配，Y=0 则代表不匹配，m 为设定的阈值，N 为样本个数。

​		这里设置了一个阈值ｍargin，表示我们只考虑不相似特征欧式距离在０～ｍargin之间的，当距离超过ｍargin的，则把其loss看做为０（即不相似的特征离的很远，其loss应该是很低的；而对于相似的特征反而离的很远，我们就需要增加其loss，从而不断更新成对样本的匹配程度）。





## OpenPose（关键点）

[Realtime Multi-Person 2D Human Pose Estimation using Part Affinity Fields](https://www.cnblogs.com/taoshiqian/p/9335525.html)

​		能有效检测图像中多个人的2D姿态。使用PAFs (Part Affinity Fields，关键点亲和场，Affinity指两点配对的关联程度，表示关键点配对的置信程度)，来学习关键点和肢体。这种结构对global context（全局上下文）进行编码，自下而上进行解析。特点：多人，高精度，实时。通过序列结构神经网络的两个分支来联合学习：关键点位置、关键点之间的联系即PAF。

<p align="center">
	<img width=60% height=60% src="Pipeline of OpenPose.png"/>  
</p>

<p align="center">
	<img width=60% height=60% src="Architecture of OpenPose.png"/>  
</p>

#### loss

<p align="center">
	<img width=40% src="loss in open-pose.png"/>  
</p>

​	W(p)是一个二值掩码（取值0或1），当关键点不存在或者标签丢失时值为0，防止惩罚了true positive predictions。

#### 关键点热力图

- 维度

  <p align="center">
  	<img width=20% src="cmp heatmap.png"/>  
  </p>

- heatmap的生成

  <p align="center">
  	<img width=90% src="cmp loss in open-pose.png"/>  
  </p>

  如果存在多个目标，则热力图中每个位置的值设置为**所有人中该关键点在该位置的最大值**。

#### Part Affinity Fields(部分亲和度向量场)

<p align="center">
	<img width=40% src="paf in open-pose.jpg"/>  
</p>

​		PAFs是一个2D矢量场，保留了位置和方向。表示一个肢体，由两个关键点构成，在肢体上的每个点是从一个关键点到下一个关键点的2D单位矢量。 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bj_1%2Ck%7D%2Cx_%7Bj_2%2Ck%7D) 表示ground truth中的关键点 *j1，j2* 的坐标，这两个关键点组成第 *k* 号人的一个肢体 *c*。如果一个点P在这个肢体上面，则P的值为 *j1* 指向 *j2* 的单位矢量；其他点都是零向量，如公式：

<p align="center">
	<img width=60% src="paf loss in open-pose.png"/>  
</p>

​		肢体 *c* 上的点满足：

<p align="center">
	<img width=50% src="paf loss a in open-pose.png"/>  
</p>

其中：

1. <p align="left">
   <img width=50% src="paf loss b in open-pose.png"/>  
   </p>

2. δ表示肢体的宽度

   若某个像素点存在多个肢体，则取平均值：

   <p align="center">
   	<img width=90% src="paf loss c in open-pose.png"/>  
   </p>

​        在测试阶段，关键点 *dj1,dj2* 和 PAF 沿着“关键点对”组成的线段计算上面 PAF 的线积分来度量这对“关键点对”的关联程度（亲和度）： 

<p align="center">
	<img width=80% src="paf loss d in open-pose.png"/>  
</p>

<p align="center">
	<img width=30% src="paf loss e in open-pose.png"/>  
</p>

​        在实践中，通过对 *u* 的均匀间隔采样来近似求积分。



## [UNet++](https://zhuanlan.zhihu.com/p/44958351)

<p align="center">
	<img width=100% height=100% src="UNet++.png"/>  
</p>

​		文章改进skip connection，并引入deep supervision的思路。网络的loss函数是由不同层得到的分割图的loss的平均，每层的loss函数为DICE LOSS和Binary cross-entropy LOSS之和。作者认为引入DSN（deep supervision net）后，通过model pruning（模型剪枝）能够实现模型的两种模式：高精度模式和高速模式。



## Unet

<img src="image-20200102111045518.png" alt="image-20200102111045518" style="zoom:50%;" />

- Data Augmentation

  将训练样本进行随机弹性形变是训练分割网络的关键。我们使用随机位移矢量在粗糙的3*3网格上(random displacement vectors on a coarse 3 by 3 grid)产生平滑形变(smooth deformations)。位移是从10像素标准偏差的高斯分布中采样的。然后使用bicubic插值计算每个像素的位移。在contracting path的末尾采用drop-out 层更进一步增加数据。

  <img src="https://img-blog.csdn.net/20180223105910103?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvamlhbnl1Y2hlbjIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="这里写图片描述" style="zoom:50%;" />

  <img src="https://img-blog.csdn.net/2018022310592540?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvamlhbnl1Y2hlbjIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="这里写图片描述" style="zoom:50%;" />

- 为了预测图像边界区域的像素点，我们采用镜像图像的方式补全缺失的环境像素。

### FCN VS Unet

- add vs concat

  在相加的方式下，feature map 的维度没有变化，但每个维度都包含了更多特征，对于普通的分类任务这种不需要从 feature map 复原到原始分辨率的任务来说，这是一个高效的选择；而**拼接则保留了更多的维度/位置信息**，这使得后面的 layer 可以在浅层特征与深层特征自由选择，这对语义分割任务来说更有优势。

  

## Meta Learning



## CenterNet



## YOLOv4

