# -*- coding: utf-8 -*-
"""阶段2 入口: 必须用 D:\\vllm\\python\\python.exe 2_对齐微调.py 运行。
改实验设置去 config2.py。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_stage2 import run

if __name__ == "__main__":
    run()
