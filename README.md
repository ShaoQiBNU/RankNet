
RankNet算法介绍
===============

# 一. 简介

> RankNet是2005年微软提出的一种pairwise的Learning to Rank算法，它从概率的角度来解决排序问题。RankNet的核心是**提出了一种概率损失函数来学习Ranking Function**，并应用Ranking Function对文档进行排序。这里的Ranking Function可以是任意对参数可微的模型，也就是说，该概率损失函数并不依赖于特定的机器学习模型，在论文中，RankNet是基于神经网络实现的。除此之外，GDBT等模型也可以应用于该框架。

# 二. 算法过程

## (一) 相关性概率

> 定义两个概率：预测相关性概率和真实相关性概率如下：

### 1. 预测相关性概率

> 对于任意一个pair的doc对<a href="https://www.codecogs.com/eqnedit.php?latex=(U_i,&space;U_j)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?(U_i,&space;U_j)" title="(U_i, U_j)" /></a>，模型输出的score为<a href="https://www.codecogs.com/eqnedit.php?latex=(s_i,&space;s_j)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?(s_i,&space;s_j)" title="(s_i, s_j)" /></a>，那么根据模型的预测，<a href="https://www.codecogs.com/eqnedit.php?latex=U_i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_i" title="U_i" /></a>比<a href="https://www.codecogs.com/eqnedit.php?latex=U_j" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_j" title="U_j" /></a>与query更相关的概率定义如下：
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=P_{ij}&space;=&space;P(U_i>U_j)&space;=&space;\frac{1}{1&plus;e^{-\sigma&space;(s_i-s_j)}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{ij}&space;=&space;P(U_i>U_j)&space;=&space;\frac{1}{1&plus;e^{-\sigma&space;(s_i-s_j)}}" title="P_{ij} = P(U_i>U_j) = \frac{1}{1+e^{-\sigma (s_i-s_j)}}" /></a>

> 由于RankNet使用的模型一般为神经网络，根据经验sigmoid函数能提供一个比较好的概率评估。参数σ决定sigmoid函数的形状，对最终结果影响不大。

> RankNet有一个结论：对于任何一个长度为n的排列，只需要知道n-1个相邻item的概率<a href="https://www.codecogs.com/eqnedit.php?latex=P_{i,i&plus;1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{i,i&plus;1}" title="P_{i,i+1}" /></a> ，不需要计算所有的pair，就可以推断出来任何两个item的排序概率，从而减少了计算量。例如已知<a href="https://www.codecogs.com/eqnedit.php?latex=P_{i,k}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{i,k}" title="P_{i,k}" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=P_{k,j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{k,j}" title="P_{k,j}" /></a>，则<a href="https://www.codecogs.com/eqnedit.php?latex=P_{i,j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{i,j}" title="P_{i,j}" /></a>可通过下面的过程推导得出：
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=\begin{matrix}&space;\\&space;P_{i,j}=\frac{1}{1&plus;e^{-\sigma&space;(s_i-s_j)}}&space;\\&space;\\&space;=\frac{1}{1&plus;e^{-\sigma&space;(s_i-s_k&plus;s_k-s_j)}}&space;\\&space;\\&space;=\frac{1}{1&plus;e^{-\sigma&space;(s_i-s_k)}\cdot&space;e^{-\sigma&space;(s_i-s_k)}}&space;\\&space;\\&space;=\frac{e^{\sigma&space;(s_i-s_k)}\cdot&space;e^{\sigma&space;(s_i-s_k)}}{1&plus;e^{\sigma&space;(s_i-s_k)}\cdot&space;e^{\sigma&space;(s_i-s_k)}}&space;\\&space;\\&space;=\frac{P_{i,k}\cdot&space;P_{k,j}}{1&plus;2P_{i,k}P_{k,j}-P_{i,k}-P_{k,j}}&space;\end{matrix}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\begin{matrix}&space;\\&space;P_{i,j}=\frac{1}{1&plus;e^{-\sigma&space;(s_i-s_j)}}&space;\\&space;\\&space;=\frac{1}{1&plus;e^{-\sigma&space;(s_i-s_k&plus;s_k-s_j)}}&space;\\&space;\\&space;=\frac{1}{1&plus;e^{-\sigma&space;(s_i-s_k)}\cdot&space;e^{-\sigma&space;(s_i-s_k)}}&space;\\&space;\\&space;=\frac{e^{\sigma&space;(s_i-s_k)}\cdot&space;e^{\sigma&space;(s_i-s_k)}}{1&plus;e^{\sigma&space;(s_i-s_k)}\cdot&space;e^{\sigma&space;(s_i-s_k)}}&space;\\&space;\\&space;=\frac{P_{i,k}\cdot&space;P_{k,j}}{1&plus;2P_{i,k}P_{k,j}-P_{i,k}-P_{k,j}}&space;\end{matrix}" title="\begin{matrix} \\ P_{i,j}=\frac{1}{1+e^{-\sigma (s_i-s_j)}} \\ \\ =\frac{1}{1+e^{-\sigma (s_i-s_k+s_k-s_j)}} \\ \\ =\frac{1}{1+e^{-\sigma (s_i-s_k)}\cdot e^{-\sigma (s_i-s_k)}} \\ \\ =\frac{e^{\sigma (s_i-s_k)}\cdot e^{\sigma (s_i-s_k)}}{1+e^{\sigma (s_i-s_k)}\cdot e^{\sigma (s_i-s_k)}} \\ \\ =\frac{P_{i,k}\cdot P_{k,j}}{1+2P_{i,k}P_{k,j}-P_{i,k}-P_{k,j}} \end{matrix}" /></a>

> 最后一步的推导中，使用了<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma&space;(s_i-s_j)=ln\frac{P_{i,j}}{1-P_{i,j}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\sigma&space;(s_i-s_j)=ln\frac{P_{i,j}}{1-P_{i,j}}" title="\sigma (s_i-s_j)=ln\frac{P_{i,j}}{1-P_{i,j}}" /></a>的推导关系。下图为<a href="https://www.codecogs.com/eqnedit.php?latex=P_{i,k}=P_{k,j}=P" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{i,k}=P_{k,j}=P" title="P_{i,k}=P_{k,j}=P" /></a>时，不同取值对<a href="https://www.codecogs.com/eqnedit.php?latex=P_{i,j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{i,j}" title="P_{i,j}" /></a>的影响：

![image](https://github.com/ShaoQiBNU/RankNet/blob/master/images/1.png)

> 当<a href="https://www.codecogs.com/eqnedit.php?latex=P=0" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P=0" title="P=0" /></a>时，<a href="https://www.codecogs.com/eqnedit.php?latex=P_{i,j}=0" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{i,j}=0" title="P_{i,j}=0" /></a>，表示<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>排<a href="https://www.codecogs.com/eqnedit.php?latex=U_{k}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{k}" title="U_{k}" /></a>后面，<a href="https://www.codecogs.com/eqnedit.php?latex=U_{k}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{k}" title="U_{k}" /></a>排<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>后面，则<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>一定排<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>后面；
>
> 当<a href="https://www.codecogs.com/eqnedit.php?latex=0<P<0.5" target="_blank"><img src="https://latex.codecogs.com/svg.latex?0<P<0.5" title="0<P<0.5" /></a>时，<a href="https://www.codecogs.com/eqnedit.php?latex=P_{i,j}<P" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{i,j}<P" title="P_{i,j}<P" /></a>
>
> 当<a href="https://www.codecogs.com/eqnedit.php?latex=P=0.5" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P=0.5" title="P=0.5" /></a>时，<a href="https://www.codecogs.com/eqnedit.php?latex=P_{i,j}=P=0.5" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{i,j}=P=0.5" title="P_{i,j}=P=0.5" /></a>，表示<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>有一半概率排<a href="https://www.codecogs.com/eqnedit.php?latex=U_{k}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{k}" title="U_{k}" /></a>前面，<a href="https://www.codecogs.com/eqnedit.php?latex=U_{k}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{k}" title="U_{k}" /></a>有一半概率排<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>前面，则<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>有一半概率排<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>前面；
>
> 当<a href="https://www.codecogs.com/eqnedit.php?latex=0.5<P<1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?0.5<P<1" title="0.5<P<1" /></a>时，<a href="https://www.codecogs.com/eqnedit.php?latex=P_{i,j}>P" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{i,j}>P" title="P_{i,j}>P" /></a>；
>
> 当<a href="https://www.codecogs.com/eqnedit.php?latex=P=1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P=1" title="P=1" /></a>时，<a href="https://www.codecogs.com/eqnedit.php?latex=P_{i,j}=P=1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{i,j}=P=1" title="P_{i,j}=P=1" /></a>，表示<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>排<a href="https://www.codecogs.com/eqnedit.php?latex=U_{k}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{k}" title="U_{k}" /></a>前面，<a href="https://www.codecogs.com/eqnedit.php?latex=U_{k}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{k}" title="U_{k}" /></a>排<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>前面，则<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>一定排<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>前面。

### 2. 真实相关性概率

> 训练数据中的pair的doc对<a href="https://www.codecogs.com/eqnedit.php?latex=(U_i,&space;U_j)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?(U_i,&space;U_j)" title="(U_i, U_j)" /></a>有一个关于query相关性的label，该label含义为：<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>比<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>与query更相关是否成立。因此，定义<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>比<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>更相关的真实概率如下：
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=\overline{P_{ij}}&space;=&space;\frac{1&plus;S_{ij}}{2}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\overline{P_{ij}}&space;=&space;\frac{1&plus;S_{ij}}{2}" title="\overline{P_{ij}} = \frac{1+S_{ij}}{2}" /></a>

> 如果<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>比<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>更相关，则<a href="https://www.codecogs.com/eqnedit.php?latex=S_{ij}=1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?S_{ij}=1" title="S_{ij}=1" /></a>；如果<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>不如<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>相关，则<a href="https://www.codecogs.com/eqnedit.php?latex=S_{ij}=-1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?S_{ij}=-1" title="S_{ij}=-1" /></a>；如果<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>相关程度相同，则<a href="https://www.codecogs.com/eqnedit.php?latex=S_{ij}=0" target="_blank"><img src="https://latex.codecogs.com/svg.latex?S_{ij}=0" title="S_{ij}=0" /></a>。

## (二) 损失函数

> 对于一个排序，RankNet从各个doc的相对关系来评价排序结果的好坏，排序的效果越好，那么有错误相对关系的pair就越少。所谓错误的相对关系即如果根据模型输出<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>排在<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>前面，但真实label为<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>的相关性小于<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>，那么就记一个错误pair。
>
> RankNet本质上就是以错误的pair最少为优化目标。而在抽象成cost function时，RankNet实际上是引入了概率的思想：不是直接判断<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>排在<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>前面，而是说<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>以一定的概率P排在<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>前面，即是以预测概率与真实概率的差距最小作为优化目标。最后，RankNet使用Cross Entropy作为cost function，来衡量<a href="https://www.codecogs.com/eqnedit.php?latex=P_{i,j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{i,j}" title="P_{i,j}" /></a>对<a href="https://www.codecogs.com/eqnedit.php?latex=\overline{P_{i,j}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\overline{P_{i,j}}" title="\overline{P_{i,j}}" /></a>的拟合程度，定义如下：
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=C_{i,j}=-\overline{P_{i,j}}log(P_{i,j})-(1-\overline{P_{i,j}})log(1-P_{i,j})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?C_{i,j}=-\overline{P_{i,j}}log(P_{i,j})-(1-\overline{P_{i,j}})log(1-P_{i,j})" title="C_{i,j}=-\overline{P_{i,j}}log(P_{i,j})-(1-\overline{P_{i,j}})log(1-P_{i,j})" /></a>

> 代入得：<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{matrix}&space;\\&space;C_{i,j}=-\frac{1}{2}(1&plus;S_{i,j})log\frac{1}{1&plus;e^{-\sigma&space;(s_i-s_j)}}-\frac{1}{2}(1-S_{i,j})log\frac{e^{-\sigma&space;(s_i-s_j)}}{1&plus;e^{-\sigma&space;(s_i-s_j)}}&space;\end{matrix}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\begin{matrix}&space;\\&space;C_{i,j}=-\frac{1}{2}(1&plus;S_{i,j})log\frac{1}{1&plus;e^{-\sigma&space;(s_i-s_j)}}-\frac{1}{2}(1-S_{i,j})log\frac{e^{-\sigma&space;(s_i-s_j)}}{1&plus;e^{-\sigma&space;(s_i-s_j)}}&space;\end{matrix}" title="\begin{matrix} \\ C_{i,j}=-\frac{1}{2}(1+S_{i,j})log\frac{1}{1+e^{-\sigma (s_i-s_j)}}-\frac{1}{2}(1-S_{i,j})log\frac{e^{-\sigma (s_i-s_j)}}{1+e^{-\sigma (s_i-s_j)}} \end{matrix}" /></a>
>
> 下图展示了当<a href="https://www.codecogs.com/eqnedit.php?latex=S_{i,j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?S_{i,j}" title="S_{i,j}" /></a>分别取1，0，-1时，cost function以<a href="https://www.codecogs.com/eqnedit.php?latex=s_{i}-s_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?s_{i}-s_{j}" title="s_{i}-s_{j}" /></a>为自变量的示意图。当<a href="https://www.codecogs.com/eqnedit.php?latex=S_{i,j}=1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?S_{i,j}=1" title="S_{i,j}=1" /></a>时，<a href="https://www.codecogs.com/eqnedit.php?latex=C_{i,j}=log(1&plus;e^{-\sigma&space;(s_i-s_j)})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?C_{i,j}=log(1&plus;e^{-\sigma&space;(s_i-s_j)})" title="C_{i,j}=log(1+e^{-\sigma (s_i-s_j)})" /></a>，<a href="https://www.codecogs.com/eqnedit.php?latex=s_{i}-s_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?s_{i}-s_{j}" title="s_{i}-s_{j}" /></a>越大，代价函数越小；当<a href="https://www.codecogs.com/eqnedit.php?latex=S_{i,j}=-1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?S_{i,j}=-1" title="S_{i,j}=-1" /></a>时，<a href="https://www.codecogs.com/eqnedit.php?latex=C_{i,j}=log(1&plus;e^{-\sigma&space;(s_j-s_i)})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?C_{i,j}=log(1&plus;e^{-\sigma&space;(s_j-s_i)})" title="C_{i,j}=log(1+e^{-\sigma (s_j-s_i)})" /></a>，<a href="https://www.codecogs.com/eqnedit.php?latex=s_{i}-s_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?s_{i}-s_{j}" title="s_{i}-s_{j}" /></a>越小，代价函数越小；当<a href="https://www.codecogs.com/eqnedit.php?latex=S_{i,j}=0" target="_blank"><img src="https://latex.codecogs.com/svg.latex?S_{i,j}=0" title="S_{i,j}=0" /></a>，<a href="https://www.codecogs.com/eqnedit.php?latex=s_{i}-s_{j}=0" target="_blank"><img src="https://latex.codecogs.com/svg.latex?s_{i}-s_{j}=0" title="s_{i}-s_{j}=0" /></a>时，代价函数最小。

![image](https://github.com/ShaoQiBNU/RankNet/blob/master/images/2.png)

> 该代价函数有以下两个特点：
>
> 1. 当两个相关性不同的文档算出来的模型分数相同时，损失函数的值大于0，仍会对这对pair做惩罚，使他们的排序位置区分开。
>
> 2. 损失函数是一个类线性函数，可以有效减少异常样本数据对模型的影响，因此具有鲁棒性。

> 总代价函数定义如下：
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=C=\sum_{(i,j)\epsilon&space;I}^{&space;}C_{i,j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?C=\sum_{(i,j)\epsilon&space;I}^{&space;}C_{i,j}" title="C=\sum_{(i,j)\epsilon I}^{ }C_{i,j}" /></a>
>
> 其中，<a href="https://www.codecogs.com/eqnedit.php?latex=I" target="_blank"><img src="https://latex.codecogs.com/svg.latex?I" title="I" /></a>表示在同一query下，所有pair对，每个pair有且仅有一次。

## (三) 训练和求算

> 得到可微的代价函数后，可以用随机梯度下降法来迭代更新模型参数w，对于每一对pair进行一次权重的更新。加速训练的方法是对**同一个query下的所有文档pair**全部带入神经网络进行前向预测，然后计算总差分并进行误差后向反馈，这样将大大减少误差反向传播的次数。

# 四. 代码

> 构建3层神经网络实现RankNet，生成的数据特征维数为10，lable为前5个维度的特征 * 2 + 后五个维度的特征 * 3得到，代码如下：

```python
# coding=utf-8
###################### load packages #########################
import tensorflow as tf
import numpy as np
import random

###################### 构造训练数据 #########################
def get_train_data(batch_size=32):
    # 生成的数据特征维数为10,lable为前5个维度的特征 * 2 + 后五个维度的特征 * 3得到
    X1, X2 = [], []
    Y1, Y2 = [], []

    for i in range(0, batch_size):
        x1 = []
        x2 = []
        o1 = 0.0
        o2 = 0.0
        for j in range(0, 10):
            r1 = random.random()
            r2 = random.random()
            x1.append(r1)
            x2.append(r2)

            mu = 2.0
            if j >= 5: mu = 3.0
            o1 += r1 * mu
            o2 += r2 * mu
        X1.append(x1)
        Y1.append([o1])
        X2.append(x2)
        Y2.append([o2])

    return ((np.array(X1), np.array(X2)), (np.array(Y1), np.array(Y2)))

###################### 网络结构 #########################
feature_num = 10
h1_num = 10

with tf.name_scope("input"):
    x1 = tf.placeholder(tf.float32, [None, feature_num], name="x1")
    x2 = tf.placeholder(tf.float32, [None, feature_num], name="x2")

    o1 = tf.placeholder(tf.float32, [None, 1], name="o1")
    o2 = tf.placeholder(tf.float32, [None, 1], name="o2")

# 添加隐层节点
with tf.name_scope("layer1"):
    with tf.name_scope("w1"):
        w1 = tf.Variable(tf.random_normal([feature_num, h1_num]), name="w1")
        tf.summary.histogram("layer1/w1", w1)
    with tf.name_scope("b1"):
        b1 = tf.Variable(tf.random_normal([h1_num]), name="b1")
        tf.summary.histogram("layer1/b1", b1)

    # 此处没有添加激活函数
    with tf.name_scope("h1_o1"):
        h1_o1 = tf.matmul(x1, w1) + b1
        tf.summary.histogram("h1_o1", h1_o1)

    with tf.name_scope("h2_o1"):
        h1_o2 = tf.matmul(x2, w1) + b1
        tf.summary.histogram("h2_o1", h1_o2)

# 添加输出节点
with tf.name_scope("output"):
    with tf.name_scope("w2"):
        w2 = tf.Variable(tf.random_normal([h1_num, 1]), name="w2")
        tf.summary.histogram("output/w2", w2)

    with tf.name_scope("b2"):
        b2 = tf.Variable(tf.random_normal([1]))
        tf.summary.histogram("output/b2", b2)

    h2_o1 = tf.matmul(h1_o1, w2) + b2
    h2_o2 = tf.matmul(h1_o2, w2) + b2


###################### loss #########################
with tf.name_scope("loss"):

    ############### 相关性概率 ###############
    ########## 真实相关性概率 #########
    o12 = o1 - o2
    lable_p = 1 / (1 + tf.exp(-o12))

    ########## 预测相关性概率 #########
    h_o12 = h2_o1 - h2_o2
    pred = 1 / (1 + tf.exp(-h_o12))

    ################## 交叉熵 ################
    cross_entropy = -lable_p * tf.log(pred) - (1 - lable_p) * tf.log(1 - pred)
    reduce_sum = tf.reduce_sum(cross_entropy, 1)
    loss = tf.reduce_mean(reduce_sum)
    tf.summary.scalar("loss", loss)


###################### train #########################
with tf.name_scope("train_op"):
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


###################### main #########################
with tf.Session() as sess:

    ###################### 过程写入tensorboard #########################
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/", sess.graph)

    ###################### 初始化 #########################
    init = tf.global_variables_initializer()
    sess.run(init)

    ###################### 训练 #########################
    for epoch in range(0, 3000):

        ############# 获取数据 ###########
        X, Y = get_train_data()

        ############# 训练 ################
        sess.run(train_op, feed_dict={x1: X[0], x2: X[1], o1: Y[0], o2: Y[1]})

        ############## 打印loss #############
        if epoch % 10 == 0:

            ######## 写入tensorboard ########
            summary_result = sess.run(summary_op, feed_dict={x1: X[0], x2: X[1], o1: Y[0], o2: Y[1]})
            writer.add_summary(summary_result, epoch)

            ######## loss ########
            l_v = sess.run(loss, feed_dict={x1: X[0], x2: X[1], o1: Y[0], o2: Y[1]})
            h_o12_v = sess.run(h_o12, feed_dict={x1: X[0], x2: X[1], o1: Y[0], o2: Y[1]})
            o12_v = sess.run(o12, feed_dict={x1: X[0], x2: X[1], o1: Y[0], o2: Y[1]})

            ######## print loss ########
            print("------ epoch[%d] loss_v[%f] ------ " % (epoch, l_v))
            for k in range(0, len(o12_v)):
                print("k[%d] o12_v[%f] h_o12_v[%f]" % (k, o12_v[k], h_o12_v[k]))
```

