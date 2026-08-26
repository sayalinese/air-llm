# 新架构航班延误分类项目

本项目面向航班延误二分类任务，当前主线是“阶段1三模态图表征提取 + 阶段2 Gemma-LoRA 软提示注入分类”的两阶段新架构。

隔壁 `模型搭建` 文件夹属于早期旧版本实验，当前项目说明、代码运行和论文结果整理均以本目录 `新架构` 为准。

## 项目目标

本项目的核心目标是让阶段2大模型分类方法超过阶段1 fused GAT 图模型基准。

当前关键结果：

| 方法 | 输入/机制 | Test AUC |
|---|---|---:|
| XGBoost | Frozen fused `z` | 0.6763 |
| MLP | Frozen fused `z` | 0.6678 |
| Stage-1 Fused GAT | `static / chain / network` 三图融合 | 0.6821 |
| Stage-2 Gemma-LoRA | Soft-token injected `z` | 0.6854 |

当前最好结果为 Stage-2 Gemma-LoRA，Test AUC 为 0.6854，已超过 Stage-1 Fused GAT 的 0.6821。

## 核心架构

整体流程如下：

```text
原始航班数据
  |
  v
数据创建与张量预处理
  |
  +-- 节点特征: dense 连续特征 + sparse 类别特征
  +-- 三类图边: static / chain / network
  |
  v
阶段1: TriModalGAT
  |
  +-- static GAT 分支
  +-- chain GAT 分支
  +-- network GAT 分支
  +-- 三路注意力融合
  |
  v
融合向量 z, 维度 128
  |
  v
阶段2: Gemma-LoRA
  |
  +-- z 投影为 soft tokens
  +-- soft tokens 注入 Gemma
  +-- LoRA 微调
  +-- 分类头输出延误概率
```

更完整的模型结构说明见 [架构.md](架构.md)。

## 目录结构

```text
新架构/
  data/
    2024数据/              # 原始或中间年度数据
    4月数据/               # 月度数据与表格数据
    张量预处理数据/         # 阶段1图模型使用的张量数据
    阶段2切片/             # 阶段2使用的 4k/40k/80k 预切片
  model/
    阶段1/                 # 阶段1模型权重、fused z、对比结果
    阶段2/                 # 阶段2模型权重、LoRA、history、预测与结果
  service/
    数据创建/              # 数据构建、三模态整理、预处理、切片生成
    阶段1/                 # 三模态 GAT 训练代码
    阶段2/                 # Gemma-LoRA 软提示注入训练代码
  架构.md                  # 模型架构详细说明
  README.md                # 项目入口说明
```

## 数据说明

当前数据分为两类：

1. 月度数据：用于当前主要实验和快速验证。
2. 年度数据：用于更大规模实验或后续扩展。

阶段2当前主要使用已经切好的数据：

| 切片 | 路径 | 用途 |
|---|---|---|
| 4K | `data/阶段2切片/4k` | 快速消融与调试 |
| 40K | `data/阶段2切片/40k` | 当前主实验 |
| 80K | `data/阶段2切片/80k` | 数据规模扩展实验 |

当前主实验口径：

| split | 来源 | 数量 |
|---|---|---:|
| train | `data/阶段2切片/40k/train.pt` | 40,000 |
| val | `data/阶段2切片/40k/val.pt` | 20,000 |
| test | `model/阶段1/fused_test.pt` | 94,446 |

阶段2切片来自阶段1导出的融合表征 `z`，不是从原始数据重新抽取特征。也就是说，阶段2和冻结表征分类器对照都建立在 Stage-1 fused representation 之上。

## 阶段1: 三模态 GAT

阶段1代码位于 `service/阶段1`，核心模型为 `TriModalGAT`。

阶段1输入包括：

| 输入 | 说明 |
|---|---|
| dense 特征 | 连续数值特征 |
| sparse 特征 | 类别特征，经 Embedding 编码 |
| static 图 | 静态关系图 |
| chain 图 | 航班链式关系图 |
| network 图 | 网络传播关系图 |

阶段1会分别训练：

```text
static
chain
network
fused
```

其中 `fused` 模式通过注意力机制融合三类图分支，并导出阶段2所需的融合向量：

```text
model/阶段1/fused_train.pt
model/阶段1/fused_val.pt
model/阶段1/fused_test.pt
```

阶段1当前测试结果：

| 模型 | Test AUC | Test PR-AUC | F1 |
|---|---:|---:|---:|
| Static GAT | 0.6761 | 0.3245 | 0.3760 |
| Chain GAT | 0.6774 | 0.3246 | 0.3785 |
| Network GAT | 0.6729 | 0.3176 | 0.3743 |
| Fused GAT | 0.6821 | 0.3355 | 0.3808 |

## 阶段2: Gemma-LoRA 软提示注入

阶段2代码位于 `service/阶段2`，入口脚本为：

```powershell
D:\vllm\python\python.exe service\阶段2\2_对齐微调.py
```

阶段2核心思路：

1. 读取阶段1导出的 `z`。
2. 将 `z` 投影为若干个 soft tokens。
3. 将 soft tokens 注入 Gemma 文本层。
4. 通过 LoRA 微调 Gemma language model 的部分线性模块。
5. 读取 soft token hidden states 并完成二分类。

当前主配置：

| 配置项 | 当前值 |
|---|---:|
| 训练切片 | 40K |
| Gemma | Gemma-4-E2B |
| `z` 维度 | 128 |
| soft token 数量 | 4 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| batch size | 16 |
| learning rate | 2e-4 |
| readout mode | `soft_prefix` |

当前最好阶段2结果：

| 指标 | 数值 |
|---|---:|
| Best Val AUC | 0.6463 |
| Test AUC | 0.6854 |
| Best epoch | 3 |
| Best global step | 7500 |

结果文件位于：

```text
model/阶段2/result_40k_软提示前置均值_训练模式修复.json
model/阶段2/history_40k_软提示前置均值_训练模式修复.csv
model/阶段2/test_predictions_40k_软提示前置均值_训练模式修复.csv
model/阶段2/stage2_fuse_40k_软提示前置均值_训练模式修复.pt
model/阶段2/gemma_lora_40k_软提示前置均值_训练模式修复/
```

## 运行流程

如果从数据开始完整重跑，建议按下面顺序执行。

### 1. 构建年度或月度基础数据

```powershell
D:\vllm\python\python.exe service\数据创建\1_构建2024年数据.py
```

### 2. 整理三模态数据

```powershell
D:\vllm\python\python.exe service\数据创建\2_三模态数据整理.py
```

### 3. 生成阶段1张量预处理数据

```powershell
D:\vllm\python\python.exe service\数据创建\3_数据预处理.py
```

### 4. 训练阶段1三模态 GAT

```powershell
D:\vllm\python\python.exe service\阶段1\1_提取器预训练.py
```

该步骤会生成阶段1权重、阶段1对比表，以及阶段2需要的 `fused_train.pt`、`fused_val.pt`、`fused_test.pt`。

### 5. 生成阶段2切片

```powershell
D:\vllm\python\python.exe service\数据创建\4_生成数据划分切片.py
```

该步骤会生成：

```text
data/阶段2切片/4k
data/阶段2切片/40k
data/阶段2切片/80k
```

### 6. 训练阶段2 Gemma-LoRA

```powershell
D:\vllm\python\python.exe service\阶段2\2_对齐微调.py
```

阶段2实验配置集中在：

```text
service/阶段2/config2.py
```

如需切换 4K、40K、80K 或全量训练，优先修改 `config2.py` 中的配置项。

## 重要注意事项

1. 当前项目主线是 `新架构`，隔壁 `模型搭建` 是旧版本实验，暂时不作为当前主结果来源。
2. 阶段2使用的 4K、40K、80K 是预先切好的数据，目录为 `data/阶段2切片`。
3. 阶段2输入 `z` 来自 Stage-1 Fused GAT，不是原始表格特征。
4. 冻结表征上的 XGBoost、MLP 等基础模型是 probe 对照，用于说明阶段2不是简单分类头替换。
5. 阶段2曾修复过训练/验证模式切换问题：验证后需要恢复训练模式，否则 LoRA dropout 会被关闭。
6. 当前主指标为 ROC-AUC；F1、Precision、Recall 需要明确阈值来源，正式表格应优先使用验证集选择阈值后再评估测试集。

## 论文表述建议

可以将当前模型概括为：

> 本文提出一种面向航班延误预测的两阶段混合架构。第一阶段通过三模态 GAT 建模航班的静态关系、链式关系与网络传播关系，并学习融合表征；第二阶段将该融合表征投影为软提示 token 注入 Gemma 大模型，并通过 LoRA 微调完成延误分类。实验结果表明，Stage-2 Gemma-LoRA 在全量测试集上取得 0.6854 的 AUC，超过 Stage-1 Fused GAT 的 0.6821。

推荐表名：

```text
冻结融合表征分类器与两阶段模型对比
```

推荐方法名：

```text
Stage-2 Gemma-LoRA with Soft-Token Injection
```
