# -*- coding: utf-8 -*-
"""把航班字段变成 prompt 文本, 供 Gemma 语义编码。"""
import pandas as pd


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
    return dict(zip(df['航班ID'], s))
