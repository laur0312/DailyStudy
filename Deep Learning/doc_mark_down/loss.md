---
typora-root-url: ..\img_loss
---

## LOSS

[目标检测回归损失函数](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247493294&idx=1&sn=d64822f1c2ca25901f7b707d78028364&chksm=ec1c0b57db6b82414c9177a13da5c1cceda8963785e22cb777b43892608c9ce71afc90c8d0c1&scene=21#wechat_redirect)

- smmoth-L1/L1/L2
- IOU/GIOU/DIOU/CIOU



## [语义分割的loss盘点](https://mp.weixin.qq.com/s/ra2qpFSbSuuJPDj39A5MWA)

<p align="center">
	<img width=80% src="loss in segmentation.jpg"/>  
</p>

**交叉熵Loss**

<p align="center">    
    <img width=30% src="log loss.svg"/>  
</p>

```python
#二值交叉熵，这里输入要经过sigmoid处理
import torch
import torch.nn as nn
import torch.nn.functional as F
nn.BCELoss(F.sigmoid(input), target)
#多分类交叉熵, 用这个 loss 前面不需要加 Softmax 层
nn.CrossEntropyLoss(input, target)
```



**带权交叉熵 Loss**

<p align="center">    
    <img width=30% src="weighted log loss.svg"/>  
</p>

​		其中

<p align="center">    
    <img width=15% src="weighted.svg"/>  
</p>

**Focal Loss**

​		Focal Loss来解决**难易样本数量不平衡**，易分样本（即，置信度高的样本）对模型的提升效果非常小，模型应该主要关注与那些难分样本。损失函数训练的过程中关注的样本优先级就是**正难 > 负难 > 正易 > 负易**。目前**在图像分割上只是适应于二分类**。

<p align="center">    
    <img width=80% src="focal loss.webp"/>  
</p>

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
```

**Dice Loss**

<p align="center">    
    <img width=30% src="dice loss.svg"/>  
</p>

​		**不均衡的场景下的确好使**。有时使用dice loss会使训练曲线有时不可信，而且dice loss好的模型并不一定在其他的评价标准上效果更好，不可信的原因是梯度，对于softmax或者是log loss其梯度简化而言为 *p−t*，*t*为目标值，*p*为预测值。而dice loss为：

<p align="center">    
    <img width=10% src="grad loss in dice loss.jpg"/>  
</p>

​		如果*p*、*t*过小则会导致梯度变化剧烈，导致训练困难。**Dice loss，对小目标是十分不利的**，因为在只有前景和背景的情况下，小目标一旦有部分像素预测错误，那么就会导致Dice大幅度的变动，从而导致梯度变化剧烈，训练不稳定。

```python
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        num = targets.size(0)
        // 为了防止除0的发生
        smooth = 1
        
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
 
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
```

**Tversky Loss**

​		论文地址为：`https://arxiv.org/pdf/1706.05721.pdf`。实际上Dice Loss只是Tversky loss的一种特殊形式而已，我们先来看一下Tversky系数的定义，它是Dice系数和Jaccard系数（即IOU系数）的广义系数，公式为：

<p align="center">
	<img width=30% src="Tversky loss.svg"/>  
</p>

​		这里A表示预测值而B表示真实值。其中|A-B|代表FP（假阳性），|B-A|代表FN（假阴性），通过调整alpha和beta这两个超参数可以控制这两者之间的权衡，进而影响召回率等指标。

```python
def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)
```

**Dice + Focal loss**

​		Dice + Focal loss来处理小器官的分割问题。在前面的讨论也提到过，直接使用Dice会使训练的稳定性降低，而此处再添加上Focal loss这个神器。

<p align="center">
	<img width=100% src="dice + focal loss.jpg"/>  
</p>

