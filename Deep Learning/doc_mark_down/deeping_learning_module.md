---
typora-root-url: ..\img_deep_learning_module
---



## BatchNorm and Dropout

**如何通过方差偏移理解批归一化与Dropout之间的冲突**

Dropout 与 BN 之间冲突的关键是网络状态切换过程中存在神经方差的（neural variance）不一致行为。试想若有图一中的神经响应 X，当网络从训练转为测试时，Dropout 可以通过其随机失活保留率（即 p）来缩放响应，并在学习中改变神经元的方差，而 BN 仍然维持 X 的统计滑动方差。这种方差不匹配可能导致数值不稳定（见下图中的红色曲线）。而随着网络越来越深，最终预测的数值偏差可能会累计，从而降低系统的性能。简单起见，作者们将这一现象命名为**<font color=green size=3>「方差偏移」</font>**。事实上，如果没有 Dropout，那么实际前馈中的神经元方差将与 BN 所累计的滑动方差非常接近（见下图中的蓝色曲线），这也保证了其较高的测试准确率。

<p align="center">
	<img width=80% height=80% src="BN and Dropout.png"/>  
</p>

作者采用了两种策略来探索如何打破这种局限。一个是在所有 BN 层后使用 Dropout，另一个就是修改 Dropout 的公式让它对方差并不那么敏感，就是**<font color=green size=3>「高斯Dropout」</font>**。

- 第一个方案比较简单，把Dropout放在所有BN层的后面就可以了，这样就不会产生方差偏移的问题，但实则有逃避问题的感觉。
- 第二个方案来自Dropout原文里提到的一种高斯Dropout，是对Dropout形式的一种拓展。作者进一步拓展了高斯Dropout，提出了一个均匀分布Dropout，这样做带来了一个好处就是这个形式的Dropout（又称为“Uout”）对方差的偏移的敏感度降低了，总得来说就是整体方差偏地没有那么厉害了。



## Normalization

我们将输入的 feature map shape 记为[N, C, H, W]，其中N表示batch size，即N个样本；C表示通道数；H、W分别表示特征图的高度、宽度。这几个方法主要的区别就是在：

1. BN是在batch上，对N、H、W做归一化，而保留通道 C 的维度。BN对较小的batch size效果不好。**<font color=red size=3>BN适用于固定深度的前向神经网络，如CNN，不适用于RNN；</font>**

2. LN在通道方向上，对C、H、W归一化，**<font color=red size=3>主要对RNN效果明显；</font>**

3. IN在图像像素上，对H、W做归一化，**<font color=red size=3>用在风格化迁移；</font>**

4. GN将channel分组，然后再做归一化；

   <p align="center">
   	<img width=100% height=100% src="Normalization.webp"/>  
   </p>

**BN 和 IN 可以**设置参数：`momentum`和`track_running_stats`来获得在**整体数据上更准确的均值和标准差**。**LN 和 GN 只能计算当前 batch 内数据的真实均值和标准差**。

### [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf)

- **为什么要进行BN呢？**

（1）在深度神经网络训练的过程中，通常以输入网络的每一个mini-batch进行训练，这样每个batch具有不同的分布，使模型训练起来特别困难。

（2）Internal Covariate Shift (ICS) 问题：在训练的过程中，激活函数会改变各层数据的分布，随着网络的加深，这种改变（差异）会越来越大，使模型训练起来特别困难，收敛速度很慢，会出现梯度消失的问题。

- **BN的主要思想**

针对每个神经元，**使数据在进入激活函数之前，沿着通道计算每个batch的均值、方差，‘强迫’数据保持均值为0，方差为1的正态分布**，避免发生梯度消失。具体来说，就是把第1个样本的第1个通道，加上第2个样本第1个通道 ...... 加上第 N 个样本第1个通道，求平均，得到通道 1 的均值（注意是除以 N×H×W 而不是单纯除以 N，最后得到的是一个代表这个 batch 第1个通道平均值的数字，而不是一个 H×W 的矩阵）。求通道 1 的方差也是同理。对所有通道都施加一遍这个操作，就得到了所有通道的均值和方差。

- **BN的使用位置**

全连接层或卷积操作之后，激活函数之前。

- **BN算法过程**

<p align="center">
	<img width=30% height=30% src="BN-Alg.webp"/>  
</p>



**加入缩放和平移变量的原因是：保证每一次数据经过归一化后还保留原有学习来的特征，同时又能完成归一化操作，加速训练。** 这两个参数是用来学习的参数。

- **BN的作用**

（1）允许较大的学习率；

（2）减弱对初始化的强依赖性；

（3）保持隐藏层中数值的均值、方差不变，让数值更稳定，为后面网络提供坚实的基础；

（4）有轻微的正则化作用（相当于给隐藏层加入噪声，类似Dropout）

- **BN存在的问题**

（1）每次是在一个batch上计算均值、方差，如果**batch size太小**，则计算的均值、方差不足以代表整个数据分布。

（2）**batch size太大**：会超过内存容量；需要跑更多的epoch，导致总训练时间变长；会直接固定梯度下降的方向，导致很难更新。

### [Layer Normalization](https://arxiv.org/pdf/1607.06450v1.pdf)

Layer Normalization (LN) 的一个优势是**不需要批训练，在单条数据内部就能归一化**。LN不依赖于batch size和输入sequence的长度，因此可以用于batch size为1和RNN中。LN用于RNN效果比较明显，但是在CNN上，效果不如BN。

### [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf)

IN针对图像像素做normalization，最初用于图像的风格化迁移。在图像风格化中，生成结果主要依赖于某个图像实例，feature map 的各个 channel 的均值和方差会影响到最终生成图像的风格。所以对整个batch归一化不适合图像风格化中，因而对H、W做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。

### [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf)

**GN是为了解决BN对较小的mini-batch size效果差的问题。**GN适用于占用显存比较大的任务，例如图像分割。对这类任务，可能 batch size 只能是个位数，再大显存就不够用了。而当 batch size 是个位数时，BN 的表现很差，因为没办法通过几个样本的数据量，来近似总体的均值和标准差。GN 也是独立于 batch 的，它是 LN 和 IN 的折中。



## NMS

<p align="center">
	<img width=50% height=50% src="NMS.webp"/>  
</p>

对NMS进行分类，大致可分为以下六种，这里是依据它们在各自论文中的核心论点进行分类，这些算法可以同时属于多种类别。

1. 分类优先：传统NMS，Soft-NMS (ICCV 2017)

2. 定位优先：IoU-Guided NMS (ECCV 2018)

3. 加权平均：Weighted NMS (ICME Workshop 2017)

   <p align="center">    
       <img width=50% height=50% src="Weighted-NMS.webp"/>  
   </p>

4. 方差加权平均：Softer-NMS (CVPR 2019)

5. 自适应阈值：Adaptive NMS (CVPR 2019)

6. +中心点距离：DIoU-NMS (AAAI 2020) 

   DIoU-NMS出现于Distance-IoU一文，研究者认为若相邻框的中心点越靠近当前最大得分框*M*的中心点，则其更有可能是冗余框。

   <p align="center">
   	<img width=25% height=25% src="DIOU.webp"/>  
   </p>

**总结：**

1. **<font color=blue size=3>加权平均法通常能够稳定获得精度与召回的提升。</font>**
2. 定位优先法、方差加权平均法与自适应阈值法需要修改模型，不够灵活。
3. **<font color=blue size=3>中心点距离法可作为额外惩罚因子与其他NMS变体结合。</font>**
4. 得分惩罚法会改变box的得分，打破了模型校准机制。
5. 运算效率的低下可能会限制它们的实时应用性。



## Metric Learning

- AMSoftmax

  AMSoftmax属于Metric Learning——缩小类内距增大类间距的策略。下图形象的解释了Softmax 和 AMSoftmax的区别，Softmax能做到的只能是划分类别间的界线——绿色虚线，而AMSoftmax可以缩小类内距增大类间距，将类的区间缩小到Target region范围，同时又会产生margin大小的类间距。

<p align="center">
	<img width=60% height=60% src="AM-softmax.jpg"/>  
</p>



<p align="center">
	<img width=80% height=80% src="AM-softmax algorithm.png"/>  
</p>



## [Deformable Conv](https://zhuanlan.zhihu.com/p/62661196)

​		Deformable conv是对feature的每个位置学习一个offset。

<p align="center">
	<img width=50% src="deformable conv.jpg"/>  
</p>

<p align="center">
	<img width=50% src="deformable roi pooling.jpg"/>  
</p>



## Separable Convolution

<p align="center">
	<img width=60% src="traditional convolution.jpg"/>  
</p>

​		Separable Convolution核心思想是将一个完整的卷积运算分解为两步进行，分别为Depthwise Convolution与Pointwise Convolution。

<p align="center">
	<img width=60% src="depthwise convolution.jpg"/>  
</p>

<p align="center">
	<img width=60% src="pointwise convolution.jpg"/>  
</p>

```
参数量

N_std = 4 × 3 × 3 × 3 = 108

N_depthwise = 3 × 3 × 3 = 27
N_pointwise = 1 × 1 × 3 × 4 = 12
N_separable = N_depthwise + N_pointwise = 39
```



## Depthwise Separable Convolution

​		该结构和常规卷积操作类似，可用来提取特征，但相比于常规卷积操作，其参数量和运算成本较低。

#### 常规卷积操作		

​		对于一张5×5像素、三通道彩色输入图片（shape为5×5×3）。经过3×3卷积核的卷积层（假设输出通道数为4，则卷积核shape为3×3×3×4），最终输出4个Feature Map，如果有same padding则尺寸与输入层相同（5×5），如果没有则为尺寸变为3×3。

<img src="https://img-blog.csdn.net/20180812161250650?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3RpbnRpbmV0bWlsb3U=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="这里写图片描述" style="zoom:30%;" />

#### Depthwise Separable Convolution

​		Depthwise Separable Convolution是将一个完整的卷积运算分解为两步进行，即Depthwise Convolution与Pointwise Convolution。

##### Depthwise Convolution

​		不同于常规卷积操作，Depthwise Convolution的一个卷积核负责一个通道，一个通道只被一个卷积核卷积。上面所提到的常规卷积每个卷积核是同时操作输入图片的每个通道。

​		同样是对于一张5×5像素、三通道彩色输入图片（shape为5×5×3），Depthwise Convolution首先经过第一次卷积运算，不同于上面的常规卷积，DW完全是在二维平面内进行。卷积核的数量与上一层的通道数相同（通道和卷积核一一对应）。所以一个三通道的图像经过运算后生成了3个Feature map(如果有same padding则尺寸与输入层相同为5×5)，如下图所示。

​		Depthwise Convolution完成后的Feature map数量与输入层的通道数相同，无法扩展Feature map。而且这种运算对输入层的每个通道独立进行卷积运算，没有有效的利用不同通道在相同空间位置上的feature信息。因此需要Pointwise Convolution来将这些Feature map进行组合生成新的Feature map。

##### Pointwise Convolution

​		Pointwise Convolution的运算与常规卷积运算非常相似，它的卷积核的尺寸为 1×1×M，M为上一层的通道数。所以这里的卷积运算会将上一步的map在深度方向上进行加权组合，生成新的Feature map。有几个卷积核就有几个输出Feature map。如下图所示。

<img src="https://img-blog.csdn.net/20180812163629103?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3RpbnRpbmV0bWlsb3U=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="这里写图片描述" style="zoom:30%;" />



## 空洞卷积

#### 空洞卷积的作用

- 扩大感受野

  ​		在deep net中为了增加感受野且降低计算量，总要进行降采样(pooling或s2/conv)，这样虽然可以增加感受野，但空间分辨率降低了。为了能不丢失分辨率，且仍然扩大感受野，可以使用空洞卷积。这在检测、分割任务中十分有用。**<font color=blue size=3>一方面感受野大了可以检测分割大目标，另一方面分辨率高了可以精确定位目标。</font>**但是，**<font color=red size=3>空洞卷积不适用于分类任务。</font>**

- 捕获多尺度上下文信息

  ​		空洞卷积有一个参数可以设置dilation rate，具体含义就是在卷积核中填充dilation rate-1个0。因此，**<font color=blue size=3>当设置不同dilation rate时，感受野就会不一样，也即获取了多尺度信息。</font>**多尺度信息在视觉任务中相当重要。

#### 空洞卷积gridding问题

​		空洞卷积是存在理论问题的，**<font color=blue size=3>论文中称为gridding，其实就是网格效应/棋盘问题。</font>**因为空洞卷积得到的某一层的结果中，邻近的像素是从相互独立的子集中卷积得到的，相互之间缺少依赖。

- 局部信息丢失

  ​		由于空洞卷积的计算方式类似于棋盘格式，某一层得到的卷积结果，来自上一层的独立的集合，没有相互依赖，因此该层的卷积结果之间没有相关性，即局部信息丢失。

- 远距离获取的信息没有相关性

  ​		由于空洞卷积稀疏的采样输入信号，使得远距离卷积得到的信息之间没有相关性，影响分类结果。



## ASPP

<p align="center">
	<img width=80% src="Atrous Spatial Pyramid Pooling.png"/>  
</p>

<p align="center">
	<img width=100% src="ASPP.png"/>  
</p>

1. Atrous Spatial Pyramid Pooling 一个1x1卷积和三个3x3的采样率为rates={6,12,18}的空洞卷积，滤波器数量为256，包含BN层。
2. Image Pooling 图像级特征，引入global context information：global average pooling–>1×1 conv + bn–>bilinearly upsample。



## [Understanding Convolution for Semantic Segmentation](https://blog.csdn.net/u011974639/article/details/79460893)

- DUC

  <p align="center">
  	<img width=80% src="DUC.png"/>  
  </p>	

  ​		设计**密集上采样卷积(dense upsampling convolution, DUC)**生成预测结果，这可以捕获在双线性上采样过程中丢失的细节信息。通过学习一些系列的放大的过滤器来放大降采样的feature map到最终想要的尺寸，就是将长宽尺寸上的损失通过通道维度来弥补。假设原图大小为 *(H,W,C)*，经过ResNet后维度变为 *(h,w,c)*，其中*h=H/r、w=W/r*，通过卷积后输出feature map维度为 *(h,w,r^2\*L)*，其中 *L* 是语义分割的类别数。最后通过reshape到 *(H,W,L)* 尺寸就可以了。

  ​		不难看出，DUC的主要思想就是将整个label map划分成与输入的feature map等尺寸的子部分。所有的子部分被叠加 *r^2* 次就可以产生整个label map了。这种变化允许我们直接作用在输出的feature map上而不用像deconvolution和unpooling那样还需要一些额外的信息。

- HDC

  ​		用一系列的dilation rates（hybrid dilation convolution, HDC），而不是只用相同的rate，并且使用ResNet-101中blocks的方式连接它们。



## [ROI Pooling](https://blog.csdn.net/zjucor/article/details/79325377)

**ROI Pooling** 存在量化误差（mis-alignment）

- 将候选框边界量化为整数点坐标值。

- 将量化后的边界区域平均分割成 k x k 个单元(bin)，对每一个单元的边界进行量化。

  <p align="center">
  	<img width=80% height=80% src="ROI Pooling mis-alignment.png"/>  
  </p>

  <p align="center">
  	<img width=40% height=40% src="ROI Pooling BP.png"/> 
  </p>

  <table><tr><td bgcolor=yellow>mis-alignment对小目标的影响更加明显。</td></tr></table>

**ROI Align**

- 遍历每一个候选区域，保持浮点数边界不做量化。

- 将候选区域分割成k x k个单元，每个单元的边界也不做量化。

- 在每个单元中计算固定四个坐标位置，用双线性内插的方法计算出这四个位置的值，然后进行最大池化操作。

- 修改反向传播。

  <p align="center">
  	<img width=80% height=80% src="ROI Align.png"/>  
  </p>

  <p align="center">
  	<img width=50% height=50% src="ROI Align BP.png"/>  
  </p>



## contextual-aware strongly-supervised classification/detection/segmentation

contextual-aware：非自身信息，例如：肺分割使用征象分类的结果等

- #### [*feature-wise transformations*](https://distill.pub/2018/feature-wise-transformations/)

  - *concatenation-based conditioning*

    <img src="image-20191231133036408.png" alt="image-20191231133036408" style="zoom:100%;" />

  - *conditional biasing*

    <img src="image-20191231133212157.png" alt="image-20191231133212157" style="zoom:100%;" />

  - *conditional scaling*

    <img src="image-20191231133445966.png" alt="image-20191231133445966" style="zoom:100%;" />



## segmentation with shape priors

- #### [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://arxiv.org/pdf/1811.12231.pdf)

  - 增强Shape信息，可以提升准确率

- #### [DeepSSM: A Deep Learning Framework for Statistical Shape Modeling from Raw Images](https://arxiv.org/pdf/1810.00111.pdf)

  - A deep learning approach to extract a low-dimensional shape representation directly from 3D images, requiring virtually no parameter tuning or user assistance.
  - 涉及数据增强

- #### [SPNet: Shape Prediction using a Fully Convolutional Neural Network](http://gregslabaugh.net/publications/ArifVSNetMICCAI2018.pdf)

  - SSM(statistical shape model)、level set methods

  - We propose a novel shape predictor network for object segmentation. The proposed deep fully convolutional neural network learns to predict shapes instead of learning pixel-wise classification. The proposed network is trained with a novel loss function that **computes the error in the shape domain**.

    <img src="image-20191231142724476.png" alt="image-20191231142724476" style="zoom:80%;" />

- #### [Deep Networks with Shape Priors for Nucleus Detection](https://arxiv.org/pdf/1807.03135.pdf)

  - We develop a new approach that we call Shape Priors with Convolutional Neural Networks (SP-CNN) to perform significantly enhanced nuclei detection



典型代表AlexNet,VGGnet,GoogLeNet，ResNet，ResNeXt，Inception-ResNet，WRNnet，DenseNet，SqueezeNt、DPN

