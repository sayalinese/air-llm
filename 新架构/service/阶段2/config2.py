# -*- coding: utf-8 -*-
"""阶段2 配置: Gemma-4-E2B 软提示注入 + LoRA。路径/超参集中在此。"""
import os

BASE = r"C:\Users\16960\Desktop\期末论文\新架构"
GEMMA_PATH = r"C:\Users\16960\.cache\modelscope\hub\models\unsloth\gemma-4-E2B"
FUSED_DIR = os.path.join(BASE, "model", "阶段1")                 # 阶段1 融合向量
TAB_CSV = os.path.join(BASE, "data", "4月数据", "4月表格.csv")     # 文本化来源
MODEL2_DIR = os.path.join(BASE, "model", "阶段2")

# LoRA (Gemma4 文本层 7 类子模块)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# 训练 (顶部变量, 直接改)
MAX_SAMPLES = 40000  # 训练样本数, -1=全量(37.7万)
MAX_LEN = 64         # 文本截断长度
NUM_SOFT = 4         # 软提示 token 数
EPOCHS = 8
LR = 2e-4
BATCH = 16
VAL_EVERY = 4        # 每 epoch 验证次数 (轮内定期保存最优+早停)
PATIENCE = 3         # 验证连续不提升次数 -> 早停
VAL_CAP = 4000       # 验证集固定规模 (不随 MAX_SAMPLES 放大, 控制验证耗时)
