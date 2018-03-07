## 基于内容的推荐算法（Content-based recommendation）

### 1.Main Idea

对于用户C，推荐与用户C之前感兴趣的items相似度高的items。



### 2.建立item画像(Item profiles)

创建关于item的特征描述

e.g. moive: author, title, actor, director....

​	text: setof important words in doccument....



### 3.建立User画像(User profiles)

创建关于User的的特征描述，使用与item相同的特征，每个特征值与user的行为有关



### 4.相似度度量

* 夹角余弦

  sim(u,c) = cos<u, c> 



### 5.局限性

* 难以确定合适的特征

* 不能推荐用户兴趣以外的内容，且用户要有较为广泛的兴趣

  e.g. 有一样东西大家都说好，但某用户之前并无此类兴趣则不能推荐给该用户

* 无法对新用户进行推荐



