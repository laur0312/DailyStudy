---
typora-root-url: ..\img_attention
---

## [重新思考深度学习中的Attention机制](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247527188&idx=1&sn=32e131de5ffd739c5fdebe277f8cf4de&chksm=ec1c86eddb6b0ffbf1fa761ac6e0def3f7eab8c480280886cdab51966b3b73fc8dd00124ce67&mpshare=1&scene=1&srcid=1214Al7ofGKkC5HXqe9EDwD6&sharer_sharetime=1607987589533&sharer_shareid=5391a0c3aeea51b626750b3a54588393&key=032c59c0a43519d1b46be5c61e981a101dd09f99df6fa3c43abc59a3c3f74d303222f9362c5022e577e5c396bd00e8de54f820e00824b9ec8286cac019d4a9155bbf516c350c8bb404d1f09ef1397924c020ecc226f70e17e1fb70eee751c3c2e4ca9093f26d3679d74b2acdedeed7708579d4e17b65e0f9553f95b61121e315&ascene=1&uin=MjE5OTA4MjI0MA%3D%3D&devicetype=Windows+10+x64&version=6300002f&lang=zh_CN&exportkey=Ae%2BspNOal5kMXrun8j8IAPk%3D&pass_ticket=QyRZ%2FNetxvXDXOwK%2FR8EcaOes5JSCxIOQ4eM%2FDIsYZTseTmEdux%2Fr295%2Fly26yCd&wx_header=0)

- Attention机制的本质就是对信息进行更好地加权融合。

- 信息可以用非线性进行变换，QKV之间可以进行特征交互，可以选择加权信息的区域。

- 在加权融合的时候可以采用多种方式计算相似度

  [《Attention is All You Need》浅读（简介+代码）](https://spaces.ac.cn/archives/4765/comment-page-1)

  - 序列编码
    - RNN
    - CNN
    - Attention
  - Attention层
    - Multi-Head Attention
    - Self Attention
    - Position Embedding

<p align="center">
	<img src="position_embedding.png"/>  
</p>



## [CBAM(2018-ECCV)](https://zhuanlan.zhihu.com/p/106084464)

Convolutional Block Attention Module

<p align="center">
	<img width=60% height=60% src="Convolutional Block Attention Module.jpg"/>  
</p>

影响卷积神经网络模型性能的3个因素：深度（ResNet）、宽度（Wide-ResNet）、基数（ResNext）、注意力。

1. Channel attention module

   <p align="center">
   	<img width=60% height=60% src="Channel attention module.jpg"/>  
   </p>

   <table><tr><td bgcolor=yellow>We argue that max-pooled features which encode the degree of the most salient part can compensate the averaged-pooled features which encode global statstics softly.</td></tr></table>

2. Spatial attention modeule

   <p align="center">
   	<img width=60% height=60% src="Spatial attention module.jpg"/>  
   </p>

   1.  **在轴的方向上对不同特征图上相同位置的像素值**进行全局的MaxPooling和AvgPooling操作，分别得到两个spatial attention map并将其concatenate，shape为[2, H, W]。
   2.  再利用一个7\*7的卷积对这个feature map进行卷积，后接一个sigmoid函数。得到一个与原特征图维数相同的加上空间注意力权重的空间矩阵。（7\*7卷积核的原因是具有更大的感受野）



## Attention & Transformer

### Attention

<p align="center">
	<img width=60% height=60% src="Attention based model.jpg"/>  
</p>

1. #### Attention机制的计算流程是怎样的？

   <p align="center">
   	<img width=60% height=60% src="Attention机制的实质.jpg"/>  
   </p>

   ​		**Attention机制的实质其实就是一个寻址（addressing）的过程**，如上图所示：给定一个和任务相关的查询**Query**向量 **q**，通过计算与**Key**的注意力分布并附加在**Value**上，从而计算**Attention Value**，这个过程实际上是**Attention机制缓解神经网络模型复杂度的体现**：不需要将所有的N个输入信息都输入到神经网络进行计算，只需要从X中选择一些和任务相关的信息输入给神经网络。

   

   **注意力机制可以分为三步**

   1. **step1-信息输入**

      用**X** = [x1, · · · , xN ]表示N 个输入信息；

   2. **step2-注意力分布计算**

      令**Key**=**Value**=**X**，则可以给出注意力分布概率

      <p align="center">
      	<img width=50% height=60% src="注意力分布.png"/>  
      </p>

      **<font color=blue size=3>常见的注意力打分机制：</font>**缩放点积常用于self-attention模型

      <p align="center">
      	<img width=60% height=80% src="注意力打分机制.jpg"/>  
      </p>

   3. **step3-信息加权平均**

      ​		注意力分布概率可以解释为在上下文查询**q**时，第i个信息受关注的程度，采用一种“软性”的信息选择机制对输入信息**X**进行编码为：

      <p align="center">
      	<img width=20% src="注意力信息加权平均.png"/>  
      </p>

      ​		这种编码方式为**软性注意力机制（soft Attention）**，又可分为普通模式（**Key** == **Value** == **X**）和键值对模式（**Key！**=**Value**）。

      <p align="center">
      	<img width=100% src="软性注意力机制.jpg"/>  
      </p>

2. #### Attention机制的变种有哪些？

   1. 硬性注意力

   2. 键值对注意力

   3. 多头注意力

      ​		多头注意力（multi-head attention）是利用多个查询Q = [q1, · · · , qM]，来平行地计算从输入信息中选取多个信息。每个注意力关注输入信息的不同部分，然后再进行拼接：

      <p align="center">
      	<img width=60% src="多头注意力.png"/>  
      </p>

3. #### 一种强大的Attention机制：为什么自注意力模型（self-Attention model）在长距离序列中如此强大？

   1. 卷积或循环神经网络难道不能处理长距离序列吗？

      ​		当使用神经网络来处理一个变长的向量序列时，我们通常可以使用卷积网络或循环网络进行编码来得到一个相同长度的输出向量序列，如图所示：

      <p align="center">
      	<img width=80% src="基于卷积网络和循环网络的变长序列编码.jpg"/>  
      </p>

      ​		从上图可以看出，无论卷积还是循环神经网络其实都是对变长序列的一种“**局部编码**”：卷积神经网络显然是基于N-gram的局部编码；**<font color=green size=3>而对于循环神经网络，由于梯度消失等问题也只能建立短距离依赖。</font>**

   2. 要解决这种短距离依赖的“局部编码”问题，从而对输入序列建立长距离依赖关系，有哪些办法呢？

      <p align="center">
      	<img width=80% src="全连接模型和自注意力模型.jpg"/>  
      </p>

      ​		由上图可以看出，全连接网络虽然是一种非常直接的建模远距离依赖的模型， 但是无法处理变长的输入序列。不同的输入长度，其连接权重的大小也是不同的。

      ​		这时我们就可以利用注意力机制来“动态”地生成不同连接的权重，这就是**<font color=blue size=3>自注意力模型（self-attention model）</font>**。由于自注意力模型的权重是动态生成的，因此可以处理变长的信息序列。 

      ​		总体来说，**为什么自注意力模型（self-Attention model）如此强大**：**<font color=purple size=3>利用注意力机制来“动态”地生成不同连接的权重，从而处理变长的信息序列。</font>**

   3. 自注意力模型（self-Attention model）具体的计算流程是怎样的呢?

      1. 1. ​	给出信息输入：用X = [x1, · · · , xN ]表示N 个输入信息；通过线性变换得到为查询向量序列，键向量序列和值向量序列。**self-Attention中的Q是对自身（self）输入的变换，而在传统的Attention中，Q来自于外部。**

            <p align="center">
            	<img width=15% src="自注意力向量.png"/>  
            </p>

      <p align="center">
      	<img width=100% src="self-attention计算过程剖解.jpg"/>  
      </p>

   4. Self Attention与传统的Attention机制非常的不同

      ​		传统的Attention是基于source端和target端的隐变量（hidden state）计算Attention的，**得到的结果是源端的每个词与目标端每个词之间的依赖关系**。但Self Attention不同，**它分别在source端和target端进行**，仅与source input或者target input自身相关的Self Attention，**捕捉source端或target端自身的词与词之间的依赖关系**；然后再把source端的得到的self Attention加入到target端得到的Attention中，捕捉source端和target端词与词之间的依赖关系。因此，self Attention比传统的Attention mechanism效果要好，主要原因之一是，传统的Attention机制忽略了源端或目标端句子中词与词之间的依赖关系，相对比，**self Attention可以不仅可以得到源端与目标端词与词之间的依赖关系，同时还可以有效获取源端或目标端自身词与词之间的依赖关系。**

### [Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

<p align="center">
	<img width=50% src="transformer模型架构.png"/>  
</p>

1. Transformer Encoder

   <p align="center">
   	<img width=70% src="transformer encoder.jpg"/>  
   </p>

   - **sub-layer-1**：**multi-head self-attention mechanism**，用来进行self-attention。

   - **sub-layer-2**：**Position-wise Feed-forward Networks**，简单的全连接网络，对每个position的向量分别进行相同的操作，包括两个线性变换和一个ReLU激活输出。

     每个sub-layer都使用了残差网络

     <p align="center">
     	<img width=30% src="layer-norm for transformer.png"/>  
     </p>

2. Transformer Decoder

   <p align="center">
   	<img width=100% src="transformer decoder.jpg"/>  
   </p>

   - **sub-layer-1**：**Masked multi-head self-attention mechanism**，用来进行self-attention，与Encoder不同：由于是序列生成过程，所以在时刻 i 的时候，大于 i 的时刻都没有结果，只有小于 i 的时刻有结果，因此需要做**Mask**。
   - **sub-layer-2**：**Position-wise Feed-forward Networks**，同Encoder。
   - **sub-layer-3**：**Encoder-Decoder attention计算**。

3. Encoder-Decoder attention 与self-attention mechanism有哪些不同？

   ​		它们都使用了 multi-head计算，不过Encoder-Decoder attention采用传统的attention机制，其中的Query是self-attention mechanism已经计算出的上一时间i处的编码值，Key和Value都是Encoder的输出，这与self-attention mechanism不同。

4. multi-head self-attention mechanism具体的计算过程是怎样的？

   <p align="center">
   	<img width=100% src="multi-head self attention.jpg"/>  
   </p>

   ​		Transformer中的Attention机制由**Scaled Dot-Product Attention**和**Multi-Head Attention**组成，上图给出了整体流程。下面具体介绍各个环节：

   - **Expand**：实际上是经过线性变换，生成Q、K、V三个向量；
   - **Split heads**: 进行分头操作，在原文中将原来每个位置512维度分成8个head，每个head维度变为64；
   - **Self Attention**：对每个head进行Self Attention，具体过程和第一部分介绍的一致；
   - **Concat heads**：对进行完Self Attention每个head进行拼接；



## Seq2Seq

[全面解析RNN,LSTM,Seq2Seq,Attention注意力机制](https://zhuanlan.zhihu.com/p/135970560)

[完全解析RNN, Seq2Seq, Attention注意力机制](https://zhuanlan.zhihu.com/p/51383402)



