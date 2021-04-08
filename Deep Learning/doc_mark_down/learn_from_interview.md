[NIMA](https://zhuanlan.zhihu.com/p/33194024)

​		Google 2018提出的图像质量评估，用一个CNN去预测分数的分布，作者采用 EMD (earth mover's distance) loss 作为损失函数。本文提出的方法不仅可以为图片评分，还可以用在自动美图的pipeline里。

​		CNN最后一层 FC 层的10个 unit 分别输出该图片获得 1 ~ 10 分的概率。然后平均值和标准差这么计算

<p align="center">
	<img width=25% height=25% src="https://pic1.zhimg.com/80/v2-30eb888da915d000be46afb5284d6838_720w.jpg"/>  
</p>
<p align="center">
	<img width=30% height=30% src="https://pic4.zhimg.com/80/v2-c67a8e19f8fb98328e17ea9fc4bfccdb_720w.jpg"/>  
</p>

​		Loss Function

<p align="center">
	<img width=50% height=50% src="https://pic4.zhimg.com/80/v2-f0c98a0a2010707879db1dbb30be214b_720w.jpg"/>  
</p>

​		其中，![](https://www.zhihu.com/equation?tex=CDF_p%28k%29+%3D+%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%7Bp_%7Bsi%7D%7D) 为预测评分概率的累加值，而不是独立的预测获得每一个评分的概率，以此代替分布。在 label 中，评分越高，累计概率越大。在 predictions 中，因为用了 softmax 确保每个独立概率都大于零 (且和为1)，因此 predictions 的累加概率也能随评分单调递增。



[抠图](https://www.cnblogs.com/smartweed/p/10373565.html)

​		Alpha matting的数学模型是

<p align="center">
	<img width=20% height=20% src="https://img2018.cnblogs.com/blog/1024369/201902/1024369-20190214105948378-1190257326.jpg"/>  
</p>

<p align="center">
	<img width=80% height=80% src="https://img2018.cnblogs.com/blog/1024369/201902/1024369-20190214110220027-1410136197.jpg"/>  
</p>

​		常见的人工添加的约束条件有**三区标注图（trimap）**和**草图（scribble）**两种。2017年Xu N等 [7]提出一种基于深度学习的新抠图算法，使用的CNN模型分为两个阶段:第一阶段是深卷积编码器，将原图和对应的trimap图作为输入，预测图像的 α；第二阶段是一个小型卷积神经网络，作用是对第一个网络的输出进行精修优化。



[光流预测FlowNet](https://zhuanlan.zhihu.com/p/37736910)

<p align="center">
	<img width=50% height=50% src="https://pic3.zhimg.com/80/v2-3bb7ff9a8fcd7d84e6ae2787042ecd9e_720w.jpg"/>  
</p>



[Guide Anchoring](https://zhuanlan.zhihu.com/p/62933156)



[路径相似性描述：Fréchet distance](https://github.com/mseitzer/pytorch-fid/blob/master/tests/test_fid_score.py)

<p align="center">
	<img width=50% height=50% src="https://pic2.zhimg.com/80/97c70266326e87b25f2af1112487da11_720w.jpg"/>  
</p>

​		Fréchet distance就是狗绳距离：主人走路径A，狗走路径B，各自走完这两条路径过程中所需要的最短狗绳长度，不同于 [hausdorff distance](https://blog.csdn.net/ssdut_209/article/details/81901669)。