# -*- coding: utf-8 -*-
"""严格历史混合检索增强上下文学习训练入口。"""
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
STAGE2_DIR = os.path.join(os.path.dirname(HERE), "阶段2")
sys.path.insert(0, STAGE2_DIR)
sys.path.insert(0, HERE)

import config_rag as R
from train_rag import run


if __name__ == "__main__":
    run()
