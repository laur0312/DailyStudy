# 分类-Logistic回归
- Logistic回归不采用 MSE 作为损失函数的原因：  
	距离目标很远或很近都会使梯度为0。  
<br>
<p align="center">
	<img width=60% height=60% src="Why Logistic Regression Does Not Use Square Error As Loss.png"/>
</p>

- 生成模型和判别模型对比  
	- 生成模型会想象；而判别模型只根据训练数据决策，受 trainging data 的影响很大；  
	- 一般而言，判别模型表现优于生成模型；但是，在下述情况下生成模型则表现更好：  
		- 训练数据很少
		- 训练数据噪声多
		- 先验概率和分类可以从不同源计算获得，如：语音识别
<br>
<p align="center"> 
	<img width=60% height=60% src="Generative v.s. Discriminative.png"/>
</p>

- Logistic回归的局限性
	- 问题描述
<br>
	Logistic回归的分类结果为线性。
<br>
<p align="center">
	<img width=60% height=60% align=center src="Limitation of Logistic Regression.png"/>
</p>
	
	- 解决方案
<br>
	使用多个Logistic回归单元，前面单元扮演特征转换的角色（将原先线性不可分的特征变换为线性可分的特征），后面单元扮演特征分类的角色。
<br>
<p align="center">
	<img width=60% height=60% src="Cascading Logistic Regression Models.png"/>
</p>
