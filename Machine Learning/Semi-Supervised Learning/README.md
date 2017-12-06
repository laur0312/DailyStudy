# Semi-Supervised Learning
- For Generative Model
	- Principle  
	<p align="center">
		<img align="center" width=60% height=60% src="Semi-Supervised Generative Model Principle.png"/>
	</p>  
	- Method  
<p align="center">
	<img width=60% height=60% src="Semi-Supervised Generative Model Step.png"/>
</p>
	<br>
	同样需要对 C2 做 Step 2 处理。

- Low-density Separation
	- Self-training  
<p align="center">
	<img width=60% height=60% src="Semi-Supervised Learning Self-training.png"/>  
<.p>
	
	- Entropy-based Regularization  
	为了避免简单粗暴地将 unlabelled data 划分为某一类，Entropy-based Regularization 被引入。	
<p align="center">
	<img width=60% height=60% src="Entropy-Based Regularization.png"/>  
</p>

- Low-density Separation V.S. Generative Model
	- Low-density Separation 采用的是 Hard label，而 Generative Model 采用的是 Soft label。  
<p align="cebter">
	<img width=80% height=80% src="Hard Label V.S. Soft Label.png"/>  
</p>
以神经网络为例，使用 hard label 对会影响结果，而使用 soft label 对结果则不会有影响。

- Smoothness Assumption  
<br>
<p align="center">
	<img width=60% height=60% src="Smoothness Assumption.png"/>
</p>

	- Cluster And Then Label  
	先分类，然后统计某一分类中哪个种类的 labelled data 多，那么属于该分类的 unlabelled data 也属于该种类
<p align="center"> 
	<img width=60% height=60% src="Cluster And Then Label.png"/>
</p>

	- Graph Construction
		- The labelled data influence their neighbors
		- Propagate through the graph
<br>
<p align="center">
	<img width=60% height=60% src="Graph Construction.png"/>
</p>  
<br>
<p align="center">
	<img width=60% height=60% src="Smoothness Definition_1.png"/>
</p>
<br>
<p align="center">
	<img width=60% height=60% src="Smoothness Definition_2.png"/>
</p>
