"""3_模型训练 (链式 LSTM 官方口径): 单向 LSTM/GRU 逐位置预测航班链上每段的 DEP_DELAY>15。

前置: 先跑 0_数据整理.py 生成链张量缓存。
用法:
    python 3_模型训练.py           # 只评估验证集
    python 3_模型训练.py --test    # 同时评估测试集
或直接 .\run.ps1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from service.train_lstm import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Evaluate test split after validation.")
    return parser.parse_args()


def main():
    args = parse_args()
    train(do_test=args.test)


if __name__ == "__main__":
    main()
