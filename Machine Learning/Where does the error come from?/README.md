# 误差来源
- bias  
简单 model，对应大的 bias；复杂 model，对应小的 bias。
- variance
简单 model，对应小的 bias；复杂 model，对应大的 bias。

# 过拟合 & 欠拟合
<div align=center>
	<img width=100% height=100% src="Bias v.s. Variance.png"/>  
</div>
Bias v.s. Variance 图中，横轴表示模型的复杂程度，纵轴表示Error。

## 过拟合
- 现象  
模型在训练集上表现良好，但是在测试集表现差，此时具有 large variance。
- 应对措施  
（1）增加训练样本；（2）使用Regularization使得模型平滑，但是需要在bias和variance之间选择合适的Regularization参数。

## 欠拟合
- 现象  
模型无法在训练集上表现良好，此时具有 large bias。
- 应对措施  
重新设计模型：（1）增加更多的特征作为输入；（2）设计更加复杂的模型。

# 使用Training Set
<div align=center>
	<img width=60% height=60% src="How to Use Training Set.png"/>  
</div>
将训练集分成 *N* 份，针对每种情况下分别训练模型、计算误差，最终选取误平均误差最小的模型。

