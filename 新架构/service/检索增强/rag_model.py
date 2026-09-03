# -*- coding: utf-8 -*-
"""检索增强模型：保留阶段2投影与LoRA接口，使用末端CLS读出。"""
import os
import sys

STAGE2_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "阶段2")
sys.path.insert(0, STAGE2_DIR)

from soft_model import SoftPromptNet


class RetrievalAugmentedSoftPromptNet(SoftPromptNet):
    """soft tokens + 全部文本上下文 + 末端CLS分类。"""

    def __init__(self, H, z_dim=128):
        super().__init__(H, z_dim=z_dim, readout_mode="cls")
