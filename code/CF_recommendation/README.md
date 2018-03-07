## 协同过滤推荐算法(Collaborative Filtering)

### user-user based

#### 1.Main Idea

对于某user，寻找与该user评分相似的集合D，然后基于D来估计用户的评分



#### 2.寻找相似用户

设r~x~ 为用户的评分向量

* 余弦夹角

* 皮尔逊相关系数

  ![1520401237(1)](C:\Users\Alxe\Desktop\1520401237(1).png)

#### 3.评分预测

设集合D为K个与user最相似的users，并且都对item s作出了评分

那么对于user对item s的评分为：

* 直接平均

![微信截图_20180307134344](C:\Users\Alxe\Desktop\微信截图_20180307134344.png)

* 加权平均

![微信截图_20180307134403](C:\Users\Alxe\Desktop\微信截图_20180307134403.png)

* many tricks possible...



###item-item based

#### 1.Main Idea

对于item s寻找其他相似的items，基于相似items来估计item s的评分



#### 2.相似度量

可以使用与user-user相同的度量方式

