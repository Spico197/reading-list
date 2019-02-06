# 《大数据智能》读书笔记
---------------------------------------------



[TOC]


---------------------------------------------

《大数据智能》的层次定位应该是“通识”和“综述”之间的一本入门性或总结性读物。除前言、后记外，全书共计八个章节。本书旨在阐述数据科学在知识智能方面的几大应用和一些简略的模型介绍。书的层次使得这本书在客观上存在一些缺点，即在某些模型的叙述中，会使用一些关键步骤代替推导过程，让我觉得有点突然。由于篇幅关系，在某些方面会一笔带过。不过我们是可以接受这些缺点的，因为在内容上仍然干货满满。

## 神经网络与深度学习

神经网络是一种类似于仿生学的“连接主义”观点。其多层多“细胞”的结构在数学含义上组成了一个特殊的函数。即神经网络其实是利用凸优化的方法进行参数选优的一个特殊函数。凸优化方法在使用时需要确定`loss`函数，通常可以为均方根误差等等。凸优化的参数学习方法可以使用`梯度下降法`，具体实现时可利用`后向传播算法`计算梯度。

多层前馈神经网络：
```math
f(x) = g_{3}\left(g_{2}\left(g_{1}\left(x;w_1\right);w_2\right);w_3\right)
```

但是，这样简单的神经网络在训练时存在重要的缺点————类似随机梯度下降算法这样的参数优化算法，只能得到局部最优的解。而深度神经网络的参数空间更大，且前几层的参数梯度会比较小，这都不利于算法收敛到全局较好的解。之后，Hinton提出了一种逐层预训练的方式，才使得预训练的研究有了些眉目。

## 知识图谱

2012年5月，谷歌在它的搜索页面中首次引入“知识图谱”，即一种更加结构化、智能化的答案。

知识图谱由“实体”（entity）及其之间的关系组成，具体可形容为`(实体1, 关系, 实体2)`的三元组。

### 大规模知识库

- [Freebase](https://developers.google.com/freebase/)：谷歌收购，API现已停用。从维基百科中抽取实体关系
- [YAGO](https://www.mpi-inf.mpg.de/de/abteilungen/databases-and-information-systems/research/yago-naga/yago/)：从Wikipedia、WordNet和GeoNames中抽取实体关系，是德国马克斯·普朗克研究所发起的项目
- [Linked Open Data](http://linkeddata.org/)：W3C于2007年发起的开放互联数据项目（https://www.lod-cloud.net）
- [OpenIE](https://allenai.org/content/team/orene/etzioni-cacm08.pdf): Open Information Extraction
- [NELL](http://rtw.ml.cmu.edu/rtw/): Never-Ending Language Learning

### 构建
- 链接数据
	- 实体链接，难点：异构、冗余
- 文本信息结构化抽取
- 多源数据知识融合
	- 实体融合：不同名称、同一含义的实体融合
	- 关系融合
	- 实例融合：三元组实例的融合

### 应用
- 查询理解：直接返回查询结果而非网页、简单推理
- 自动问答
- 文档表示：以知识图谱子图的形式作为文档

### 主要技术
- 实体链指：将网页之中的内容与实体建立链接关系，有两个主要任务：`实体识别`与`实体消歧`
- 关系抽取
	- Bootstrapping：从`模板生成`->`实例抽取`迭代直至收敛
	- 句法分析
	- 标签分类
- 知识推理
	- 推理规则：Path Ranking Algorithm（Lao & Cohen 2010）
	- 基于关系的同现统计方法
	- 谓词逻辑（Predicate Logic）形式化方法
	- 马尔科夫逻辑网络（Markov Logic Network）
- 知识表示
	- 实体标签
	- 分布式表示（TransE）
	- 知识图谱补全
	- 复杂网络中的链接预测

## 大数据系统

### 并行计算
- 库
	- Pthreads：多线程
	- OpenMP：多进程
	- MPI：消息传递机制；单程序多数据或多程序多数据：在多个节点上可以运行同一份代码，也可以运行多份代码
- 支撑软件
	- 任务调度：SLURM、OpenPBS
	- 并行文件系统：PVFS、Lustre

### 云计算
- 软件即服务（SaaS）：Email、Virtual Desktop
- 平台即服务（PaaS）：Database、Web Server
- 基础架构即服务（IaaS）：Virtual Machines

### 虚拟化技术
- 传统硬件虚拟化
	- 服务器领域虚拟化：Xen、KVM
	- 桌面虚拟化：VMWare、VirtualBox
- 容器镜像：Docker

### 分布式计算系统
- Google论文
	- Google File System: Ghemawat, et al. 2003
	- MapReduce: Dean & Ghemawat 2008
	- Bigtable: Chang, et al. 2008
- Hadoop生态系统
	- HDFS：Hadoop的分布式文件系统，每块以多个副本的形式存放在多个节点
	- YARN：Hadoop资源管理和调度系统；全局资源管理器、计算节点管理器、应用程序主控、创建容器
	- HBase：Bigtable论文提出的基于列的分布式存储
	- Hive：提供类似于SQL的HiveSQL查询语言
	- Pig：提供Pig Latin脚本语言，可以用命令式编程的模式来查询数据
	- ZooKeeper：编写分布式软件所需要的常用工具，包括分布式系统的名字服务、配置管理、同步、领导者选举、消息队列、通知系统等
	- Tez：将MapReduce推广为任意的有向无环图，更一般的MapReduce
	- Storm、S4：流式处理引擎
	- Mahout：Hadoop机器学习算法库
	- Giraph：类似Google Pregel的图计算引擎，用于处理Web链接关系图
	- Sqoop：命令行工具，用于在Hadoop和传统的关系型数据库有之间传输数据
	- Flume、Kafka、Scribe：日志收集
- Spark
	- 基于内存
- 单机图计算
	- GraphChi：计算前对图的数据进行预处理，减少计算过程中访问的随机性
	- X-Stream：把计算过程中沿边传播的消息先顺序添加到缓冲区中，再通过类似于外部排序的方式把消息重新排列
	- GridGraph：边、起点、终点划分为二维的栅格，预处理时间短
	- 计算效率：GridGraph > X-Stream > GraphChi
- NoSQL
	- 基于列存储：HBase
	- 基于文档的存储：MongoDB
	- 键值对存储
		- 单机磁盘型：BerkleyDB、LevelDB
		- 单机内存型：memcached、redis
		- 分布式：Dynamo、Riak
	- 图数据库
		- Neo4j
	- 多模型：同时支持以上若干种模型：OrientDB、ArangoDB

## 智能问答

### 专家系统
- 依赖于精确组织的知识结构
- 实质上是对知识库的一种信息检索

### 问答系统的主要组成
- 问题理解
	- Who, Whom, When, Where, What, How, Why
	- 模板匹配、词法句法分析
- 知识检索
	- 非结构化信息检索
	- 结构化信息检索
- 答案生成

## 主题模型
- 从大规模甚至海量文本集合中抽取主题和主题分布
- 主题模型能够流行的重要原因就是在模型复杂性和解释性之间做了很好的折中
- 从“文档”->“词汇”中抽取“主题”

### 潜在语义分析
- 潜在语义索引（Latent Semantic Indexing，LSI），又被称之为LSA（Latent Semantic Analysis）：为了解决检索中语义不匹配的问题（歧义和多义）
- 主题模型的最基本思想
	1. 找到一系列语义“独立”的主题（在LSI中为线性无关的矢量）
	2. 将文档生成主题上的权重分布
	3. 每个主题内部，词汇可以按照与主题的相关度进行排序，进而形成主题信息的可视化理解
- 同义：LSI的本质是挖掘词汇与词汇在文档层面的共现模式
	- 如果两个词汇经常共现，那么它们很有可能具有相同的语义
	- 如果两个词汇经常与一些相同的背景词汇共现，那么它们有可能具有相同的语义
- 多义：同一个词汇在不同背景下可能具有不同的语义

### 概率主题模型
- pLSI/pLSA

### 贝叶斯主题模型
- 全贝叶斯版本的pLSI：LDA（Latent Dirichlet Allocation）by Blei, et al. 2003
- 对pLSI的改进
	- 全贝叶斯视角的模型解释
	- 提出基于变分法的模型推导方法
	- 第一次显示地提出topic model
	- 将原始pLSI中文档与文档、词与词之间的独立假设（bag-of-word 假设），使用了可交换性（可简单理解为条件独立）进行解释
- 模型求解方法
	- Gibbs采样的方法（Griffiths & Steyvers 2004）
	- 变分法EM（Blei, et al. 2003）
- 模型选择
	- 经验设定
	- 基于复杂度的确定方法
	- 使用非参数的贝叶斯方法

## 推荐系统
### 核心问题
- 预测：推断每个用户对每个物品的喜好程度
- 推荐：根据预测分值高低进行排序推荐
- 解释：使用户信服

### 推荐系统的输入
- 用户档案
- 物体档案
- 用户打分

### 推荐系统算法
- 基于人口统计学的推荐：分析年龄段等因素和物品喜好之间的关系
- 基于内容的推荐：一个用户可能会喜欢和他曾经喜欢过的物品相似的物品
- 基于协同过滤的推荐：收集用户的历史行为和偏好信息
	- 基于用户推荐
	- 基于物品推荐
	- 基于社交网络关系推荐
	- 基于模型的推荐：如贝叶斯分类、SVD矩阵分解等
- 混合型推荐系统：利用历史数据训练模型
	- 加权融合
	- 级联
### 推荐系统的评价
- RMSE：根均方误差
- MAE：平均绝对误差
- Precision, Recall, F1, ROC
- MAP：平均准确率
- NDCG：归一化折扣增益值

### 问题
- 冷启动
- 小众用户推荐
- 可解释性
- 安全性（防止恶意攻击刷分）

## 情感分析与意见挖掘

### 研究的主要问题

- 主观性分析 (主观性分类): 判断一个句子有没有表达情感或观点. 具体来说, 句子的主观性与情感色彩并不等价
- 观点挖掘: 如挖掘`情感 - 属性`对. 

在情感, 观点分析的基础上, 又衍生出:
- 垃圾观点识别
- 情感摘要
- 舆情分析

### 观点的要素
- 观点持有者 (holder): 表达观点的主体, 如发表评论的作者, 对一个事件表达同一反应的群体, 发布报告的机构等
- 观点对象或客体 (target, entity, object): 被评论或针对的对象, 如某款产品等
- 对象的属性 (attribute, aspect, feature): 对象的某个属性, 如手机的屏幕, 价格等
- 表达观点的极性 (orientation, polarity): 褒贬

可能包含的要素:
- 观点的载体 (carrier): 如文章评论, 报纸等
- 观点发表的时间

### 情感极性词典
- 英文资源
	- WordNet(Princeton)
	- SentiWordNet
	- LIWC (Linguistic Inquiry and Word Count)
	- ANEW (Affective norms for English words)
	- MPQA (Multi-Perspective Question Answering)
- 中文资源
	- 知网(HowNet, from 董振东, 董强)
	- 学生褒贬义词典(张伟等 2004)
	- 知网"情感分析用词语集"
	- 台湾大学情感词典(Ku & Chen 2007)
	- 清华大学构建的情感词典(Li & Sun 2007)
	- 北京大学情绪词典(Xu, et al. 2010)

### 属性-观点对的提取
在多数情况下, 属性词和观点词是出现在同一个句子中的, 这就是一个自然的同现关系.

- 采用词性来筛选, 建立模板
- 采用上下文熵 (Context Entropy) 的方法 (李智超 2011)
- 根据句法分析依存关系树

### 情感分析
- 无监督学习
	- 句子的情感信息完全来源于其中的情感词(或涉及的观点词等)及其在句中的地位
	- 但要保证情感词典资源的情感信息质量
- 有监督学习
	- 特征空间可选择N-Gram词袋
	- 计算权重时, 除了词次外还可以使用比例, 对数, TF-IDF的方法筛选; 或采用压缩, 提取特征值, 矩阵奇异值分解等方法减少空间维度
	- 常见分类器如朴素贝叶斯, kNN, SVM, MaxEnt等

## 社会计算

自然语言处理在社会计算学的应用中十分广泛, 我们可以通过大规模的文本分析得出语言演变的规律, 进而从社会学角度分析. 

### 面向社会媒体的NLP的使用角度
- 词汇的时空传播与演化: 在时间上观察用语变化
- 语言使用与个体差异: 人格心理学分析, 构建用户画像
- 语言使用与社会地位: 从社会地位的角度分析用语的谦卑
- 语言使用与群体分析: 在空间上分析用语的差异
 
### 应用
- 社会预测: 预测比赛结果, 政选结果, 产品销量等
- 霸凌现象的定量分析: 通过用语进行角色判断, 方便人工及时干预

## 知识的获取方式

```
graph TD
ACL-->地域
    地域-->北美-NAACL
    地域-->欧洲-EACL
ACL-->兴趣小组
    兴趣小组-->SIGDAT-EMNLP
    兴趣小组-->SIGNLL-CoNLL

ICCL-->COLING
```
- 论文查看与检索: [ACLWeb-Anthology](https://aclanthology.info/)
- 学术期刊
    - [Computational Linguistics](https://www.mitpressjournals.org/loi/coli)
    - [Transactions of ACL](https://transacl.org/)
- 更多建议查看CCF推荐目录: [link](https://www.ccf.org.cn/xspj/gyml/)
- 预发布: [arXiv](https://arxiv.org/)
