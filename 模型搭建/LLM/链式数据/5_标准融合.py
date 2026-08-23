"""6_真序列融合: 机场近窗口计划航班序列(真 token 序列)-> 冻结Gemma编码 -> 与链式LSTM门控融合。

真序列 = 目标航班起飞前、同机场同日的计划航班序列(仅静态字段, 无实际延误/时刻, 无未来泄露),
引入"机场拥堵态势"这一 15 特征之外的新信息, 并让 LLM 注意力有真序列可建模。
前置: 先跑 0_数据整理.py (生成链张量 + airseq 文本)。
自带消融: lstm_only(纯结构) vs concat/crossattn(结构+真序列), 同数据/同头, 直接判定有无小胜。
用法:
    python 6_真序列融合.py --test                       # 默认 pool 读出 (last-token)
    python 6_真序列融合.py --test --readout crossattn   # h_t 对邻居 token 跨注意力
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Evaluate test split after validation.")
    parser.add_argument("--readout", default="pool", choices=["pool", "crossattn"],
                        help="pool=last-token 池化(浅); crossattn=对邻居序列跨注意力(不池化)。")
    parser.add_argument("--modes", default=None, help="逗号分隔的消融模式子集(默认按 readout 取)。")
    args = parser.parse_args()

    if args.readout == "crossattn":
        from service.train_airseq_ca import run
        default_modes = "lstm_only,crossattn"
    else:
        from service.train_airseq import run
        default_modes = "lstm_only,concat"
    modes = [m.strip() for m in (args.modes or default_modes).split(",") if m.strip()]
    run(do_test=args.test, modes=modes)


if __name__ == "__main__":
    main()
