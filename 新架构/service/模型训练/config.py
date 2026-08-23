# -*- coding: utf-8 -*-
"""集中配置: 路径 + 超参。改文件夹名/超参只改这里。"""
import os

BASE = r"C:\Users\16960\Desktop\期末论文\新架构"
PROC_DIR = os.path.join(BASE, "data", "张量预处理数据")   # 预处理输出(你改名只改这行)
MODEL_DIR = os.path.join(BASE, "model", "阶段1")          # 训练输出

# 特征维度
DENSE_DIM = 7
SPARSE_NUM = 8

# 模型超参
EMB_DIM = 16
HIDDEN = 128   # 256 会 OOM(full-batch), 回退128
HEADS = 4
LAYERS = 2
DROPOUT = 0.1

# 训练超参
LR = 1e-3
WD = 1e-5
USE_COSINE = True   # 实验2: lr cosine 衰减开关
PATIENCE = 10
SEED = 42
