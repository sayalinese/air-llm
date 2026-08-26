# -*- coding: utf-8 -*-
"""把航班字段变成 prompt 文本, 供 Gemma 语义编码。
PROMPT_FULL=False: 基础7字段; True: 补齐15特征内剩余可用字段 (N1, 需MAX_LEN≥96)。"""
import pandas as pd

import config2 as C


def load_prompts(csv_path):
    """返回 {航班ID: prompt文本}。"""
    df = pd.read_csv(csv_path)
    s = ("承运人" + df['承运人'].astype(str)
         + " 从" + df['出发机场'].astype(str)
         + " 飞往" + df['到达机场'].astype(str)
         + " 计划飞行" + df['计划飞行时长'].astype(str) + "分钟"
         + " 出发第" + df['计划出发分钟'].astype(str) + "分钟"
         + " 温度" + df['出发地温度'].astype(str)
         + " 风速" + df['出发地风速'].astype(str))
    if C.PROMPT_FULL:   # N1: 仅总结15特征内已有字段, 无外部知识
        s = (s + " 到达第" + df['计划到达分钟'].astype(str) + "分钟"
             + " 降水" + df['出发地降水'].astype(str)
             + " 到达地温度" + df['到达地温度'].astype(str)
             + " " + df['月'].astype(str) + "月" + df['日'].astype(str) + "日"
             + " 星期" + df['星期'].astype(str))
    return dict(zip(df['航班ID'], s))
