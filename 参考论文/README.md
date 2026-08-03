# C 题方法参考论文

本目录保存 C 题路线判断中实际查阅的本地全文。这里只记录论文实际支持什么，不把论文结论扩大成当前模型已经达到的效果。

| 本地文件 | 正式题名 | 可追溯编号 | 在本项目中的用途 |
|---|---|---|---|
| `stock2018.pdf` | A Comparative Study of Pairwise Learning Methods Based on Kernel Ridge Regression | DOI: `10.1162/neco_a_01096` | 说明细胞—基因组合属于成对学习问题，给出两步核岭回归及四类新旧对象预测设置的理论背景 |
| `natarajan2014.pdf` | Inductive Matrix Completion for Predicting Gene-Disease Associations | DOI: `10.1093/bioinformatics/btu269` | 提供生命科学中的归纳冷启动例子，说明两侧资料能帮助预测训练中未出现的对象 |
| `1306.0626v1.pdf` | Provable Inductive Matrix Completion | arXiv: `1306.0626` | 给出以两侧特征构造双线性预测 `x^T W y` 的理论形式，并讨论新对象预测 |
| `基于深度学习的药物-靶标相互作用预测研究综述_刘晓光.pdf` | 基于深度学习的药物-靶标相互作用预测研究综述 | DOI: `10.11992/tis.202308024` | 作为中文研究背景，梳理药物—靶标预测中的多源数据和深度学习方法；不直接证明本题的双侧冷启动路线 |

这些论文都只能作为方法或背景依据。阶段 7 是否采用某条路线，仍由本仓库固定的三组内层冷启动筛选和一次外层官方评分决定。
