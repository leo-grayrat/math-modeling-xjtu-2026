# C 题方法参考论文

本目录保存 C 题路线判断使用的 17 篇英文核心论文和 1 篇中文背景综述。这里只记录论文实际支持什么，不把论文结论扩大成当前模型已经达到的效果。

| 本地文件 | 论文或方法 | 可追溯编号 | 在本项目中的用途 |
|---|---|---|---|
| `stock2018.pdf` | A Comparative Study of Pairwise Learning Methods Based on Kernel Ridge Regression | DOI: `10.1162/neco_a_01096` | 说明细胞—基因组合属于成对学习问题，给出两步核岭回归及四类新旧对象预测设置的理论背景 |
| `pahikkala2013.pdf` | Efficient Regularized Least-Squares Algorithms for Conditional Ranking on Relational Data | DOI: `10.1007/s10994-013-5354-7`；arXiv: `1209.4825` | 说明条件排序只比较共享同一条件对象的关系，并给出乘积核、平方回归损失和平方排序损失的高效算法 |
| `pahikkala2009.pdf` | An Efficient Algorithm for Learning to Rank from Preference Graphs | DOI: `10.1007/s10994-008-5097-z` | 给出 RankRLS 的偏好图和拉普拉斯表达，说明无需显式保存平方数量的偏好对 |
| `natarajan2014.pdf` | Inductive Matrix Completion for Predicting Gene-Disease Associations | DOI: `10.1093/bioinformatics/btu269` | 提供生命科学中的归纳冷启动例子，说明两侧资料能帮助预测训练中未出现的对象 |
| `1306.0626v1.pdf` | Provable Inductive Matrix Completion | arXiv: `1306.0626` | 给出以两侧特征构造双线性预测 `x^T W y` 的理论形式，并讨论新对象预测 |
| `ammad-ud-din2016.pdf` | 核化贝叶斯矩阵分解与通路—响应关联 | DOI: `10.1093/bioinformatics/btw433` | 说明多种核可以选择性融合，并可产生通路层面的解释 |
| `nascimento2016.pdf` | KronRLS-MKL | DOI: `10.1186/s12859-016-0890-3` | 在成对核岭中学习两侧多核权重，并验证单侧新对象预测 |
| `cichonska2018.pdf` | pairwiseMKL | DOI: `10.1093/bioinformatics/bty277` | 说明多核加权不必显式建立巨型成对矩阵，但没有直接证明双侧冷启动 |
| `journal.pcbi.1008878.pdf` | AKLIMATE | DOI: `10.1371/journal.pcbi.1008878` | 展示多组学和通路核的监督式融合；目标基因固定，不能直接替代本题模型 |
| `vbac084.pdf` | DependANT | DOI: `10.1093/bioadv/vbac084` | 提供细胞特异 PPI 网络拓扑预测基因依赖性的思路 |
| `btab137.pdf` | BioVNN | DOI: `10.1093/bioinformatics/btab137` | 展示怎样把 Reactome 层级写进模型并获得通路解释 |
| `sciadv.abh1275.pdf` | DeepDEP | DOI: `10.1126/sciadv.abh1275` | 展示多组学和基因功能指纹预测依赖性的复杂模型上限 |
| `s12920-023-01446-6.pdf` | Predicting gene knockout effects from expression data | DOI: `10.1186/s12920-023-01446-6` | 说明降噪和少量修饰基因表达在敲除效应预测中的价值 |
| `s41598-024-67023-8.pdf` | Robust evaluation of deep learning-based representation methods | DOI: `10.1038/s41598-024-67023-8` | 提醒不能只依靠一次划分和一次显著性检验判断模型优劣 |
| `2023.11.29.569083v1.full.pdf` | Revisiting DeepDEP with ridge regression | DOI: `10.1101/2023.11.29.569083` | 说明正确设置下的简单岭回归可能超过复杂深度模型 |
| `s12967-025-07501-3.pdf` | GATDep | DOI: `10.1186/s12967-025-07501-3` | 提供表达、通路状态和基因网络相结合的样本特异解释思路 |
| `journal.pcbi.1012076.pdf` | HELP | DOI: `10.1371/journal.pcbi.1012076` | 用多组学和网络嵌入预测共同与情境特异必需基因 |
| `基于深度学习的药物-靶标相互作用预测研究综述_刘晓光.pdf` | 基于深度学习的药物-靶标相互作用预测研究综述 | DOI: `10.11992/tis.202308024` | 作为中文研究背景，梳理药物—靶标预测中的多源数据和深度学习方法；不直接证明本题的双侧冷启动路线 |

这些论文都只能作为方法或背景依据。阶段 7 是否采用某条路线，仍由本仓库固定的三组内层冷启动筛选和一次外层官方评分决定。
