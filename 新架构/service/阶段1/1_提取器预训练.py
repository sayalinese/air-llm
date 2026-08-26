# -*- coding: utf-8 -*-
"""阶段1 入口: 直接运行 `python 1_提取器预训练.py`, 不带任何参数。
要改实验设置, 只改下面几个变量。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 手动变量配置
PERIOD = "month"                                  # month=4月小测试 / year=全年
MODES = ("static", "chain", "network", "fused")   # 跑哪几个模型 (融合最后跑)
EPOCHS = 60                                       # 训练轮数


from train_stage1 import run

if __name__ == "__main__":
    run(PERIOD, MODES, EPOCHS)
