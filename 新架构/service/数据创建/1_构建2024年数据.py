# -*- coding: utf-8 -*-
"""步骤1: 合并 2024 全年原始航班表 -> 新架构/data/flight_2024.csv

这是 链式/表格/图 三种模态的共同 2024 原始数据源:
  - 表格: 直接使用本表
  - 链式: 靠 TAIL_NUM (飞机尾号) 串联同一架飞机的连续航段
  - 图  : 靠 ORIGIN_INDEX/DEST_INDEX + CRS 计划时刻 建边

分月增量写入, 内存占用低。
"""
import os
import glob
import pandas as pd

SRC = r"C:\Users\16960\Desktop\期末论文\三模态数据库建立说明\scripts\Aeolus_V2\dataset\Flight_Tabular\2024"
OUT = r"C:\Users\16960\Desktop\期末论文\新架构\data\flight_2024.csv"

os.makedirs(os.path.dirname(OUT), exist_ok=True)

total = 0
first = True
for month in range(1, 13):
    mdir = os.path.join(SRC, f"{month:02d}")
    files = sorted(glob.glob(os.path.join(mdir, "*.csv")))
    mrows = 0
    for f in files:
        df = pd.read_csv(f)
        df.to_csv(OUT, mode="w" if first else "a",
                  header=first, index=False)
        first = False
        mrows += len(df)
    total += mrows
    print(f"[month {month:02d}] files={len(files)} rows={mrows}")

size_mb = os.path.getsize(OUT) / 1e6
print(f"[DONE] total_rows={total} size={size_mb:.1f}MB -> {OUT}")
