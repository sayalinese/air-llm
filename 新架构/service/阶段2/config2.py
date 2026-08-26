# -*- coding: utf-8 -*-
"""阶段2 配置: Gemma-4-E2B 软提示注入 + LoRA。路径/超参集中在此。"""
import os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEMMA_PATH = r"C:\Users\16960\.cache\modelscope\hub\models\unsloth\gemma-4-E2B"
FUSED_DIR = os.path.join(BASE, "model", "阶段1")                 # 阶段1 融合向量
TAB_CSV = os.path.join(BASE, "data", "4月数据", "4月表格.csv")     # 文本化来源
MODEL2_DIR = os.path.join(BASE, "model", "阶段2")
SLICE_DIR = os.path.join(BASE, "data", "阶段2切片")

# LoRA (Gemma4 文本层 7 类子模块)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# 训练 (顶部变量, 直接改)
MAX_SAMPLES = 40000  # 预切片支持4K/40K/80K; 全量 -1
USE_PREBUILT_SLICES = True
VAL_SLICE_TAG = "40k"  # 各训练规模固定使用同一份20K分层验证集
MAX_LEN = 64         # 文本截断长度
NUM_SOFT = 4         # 软提示 token 数
EPOCHS = 8
LR = 2e-4
BATCH = 16
VAL_EVERY = 4        # 每 epoch 验证次数 (轮内定期保存最优+早停)
VAL_STEPS = 1250     # 固定验证步数，避免随训练集规模改变checkpoint间隔
PATIENCE = 3         # 验证连续不提升次数 -> 早停 (patience=6实验证实无收益, 恢复)
FULL_PASS_BEFORE_EARLY_STOP = True  # 全量训练至少完整遍历一轮再允许早停
VAL_CAP = 20000      # 40K切片按21-25日各抽4K; 非预切片模式下作为cap
TEST_CAP = -1        # 正式评估：全量 test(94446 条)
SEED = 42
USE_POS_WEIGHT = False  # 消融A: 4K时+0.0230但40K验证为负增益(AUC/PR双降), 回退
SHUFFLE = True          # 单开曾有正增益; 每轮打乱按天分层后的训练切片
# 消融C(降LR+放宽早停)为负增益 -0.0135, 已回退: LR保持2e-4, PATIENCE保持3
PROJ_NORM = False       # E1消融: norm+attnpool组合 -0.0463, 回退关闭
POOL_ATTN = False       # E3消融: 同上, 回退关闭
HEAD_MLP = False        # E4消融: 4K时+0.0468但40K全量test未兑现(0.6667<裸配置0.6753), 回退
TOKEN_GATE = False      # S1消融v3: 门控可学但未分化(全~0.88), val-0.01, 回退关闭
PROMPT_FULL = False     # 旧软token前置读出看不到文本; 需在文本前置模式重新验证
Z_SHORTCUT = False      # N2消融: 单开+0.0131但与N3组合负交互(0.6492<N3的0.6550), 回退
USE_SCHED = False       # N3完整test低于裸配置; 先在可信抽样/读出上重建基线
READOUT_MODE = "soft_prefix"  # soft_prefix / text_then_soft / cls
RUN_TAG = "40k_软提示前置均值_训练模式修复"
SAVE_PREDICTIONS = True
