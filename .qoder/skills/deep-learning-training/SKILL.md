---
name: deep-learning-training
description: 规范深度学习模型训练的代码组织、GPU适配、断点续训和可视化流程。在创建或修改深度学习训练脚本时使用，包括LoRA微调、XGBoost、GNN、LSTM等场景。
---

# 深度学习模型训练规范

## 目录结构

```
模型名称/
├── train.py              # 训练入口（根目录，唯一的主文件）
├── service/              # 所有模块组件
│   ├── __init__.py
│   ├── config.py         # 路径、超参数（带中文注释）
│   ├── dataset.py        # 数据加载 + tokenization
│   ├── model.py          # 模型加载 + LoRA/优化器
│   ├── evaluate.py       # 评估指标（AUC/F1/阈值搜索）
│   └── plot.py           # matplotlib 可视化
└── 模型/                 # 权重、checkpoint、图表输出
    └── lora_adapter/
```

## GPU 硬件约束

- 设备: RTX 3080 20GB
- 精度: bfloat16
- batch_size 适配: 4B模型 bs=16, 7B模型 bs=8, 小模型 bs=32+
- 必须检查 `torch.cuda.is_available()`

## 训练脚本要求

1. **config.py**: 所有超参数集中管理，每个参数带中文注释说明取值理由
2. **train.py**: 只做训练循环 + 调用 service 模块，不包含模型定义
3. **进度条**: 使用 tqdm
4. **断点续训**: 每轮结束保存 checkpoint（optimizer + scheduler + epoch + history），中断后重跑自动恢复
5. **训练历史**: 每轮记录 train_loss/val_loss 到 history.json

## 评估脚本要求

1. 保存 metrics.json（acc/auc/f1/precision/recall/threshold）
2. 评估完自动调用 plot.py 生成精度柱状图
3. 支持 logprob 提取（LLM场景）或 predict_proba（传统ML场景）

## 可视化要求

训练结束自动生成：
- `loss_curve.png`: train/val 损失折线图，标注每轮 val_loss
- `metrics.png`: ACC/AUC/F1/Precision/Recall 柱状图

## 断点续训机制

```
checkpoint/
├── adapter_model.safetensors  # 模型权重
├── adapter_config.json
└── train_state.pt              # optimizer + scheduler + epoch + history
```

- 检测到 checkpoint → 从中断 epoch 继续
- 训练完成 → 自动清理 checkpoint
- train.py 中用 try-except 包裹画图调用，失败不阻断训练

## 范例参考

`模型搭建/LLM/` 目录下的完整实现：
- service/config.py: 超参数 + 中文注释
- service/dataset.py: JSONL → tokenize → Dataset
- service/model.py: Gemma4ForCausalLM + LoRA
- service/evaluate.py: logprob 提取 + 阈值搜索
- service/plot.py: matplotlib 损失曲线 + 精度柱状图
- train.py: SFT 训练循环 + 断点续训 + 自动画图
