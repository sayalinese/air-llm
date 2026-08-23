# -*- coding: utf-8 -*-
"""步骤2: 从 2024 原始表派生 三模态 CSV (字段汉化)

输出 (每个文件夹 3 个文件):
  2024数据/ : 2024表格.csv 2024链式.csv 2024图.csv   (全年, 跑完整数据)
  month数据/ : 4月表格.csv 4月链式.csv 4月图.csv     (4月, 最小测试)

模态定义:
  表格 = 汉化航班表 + 航班ID (节点特征)
  链式 = 同一飞机尾号(TAIL_NUM)同一天 按计划出发排序 的连续航段边
  图   = 静态边(同机场同天时间窗内) + 传播边(到达后窗口内下一起飞)

用法: python build_three_modality.py month | year
"""
import os
import sys
import glob
import pandas as pd

BASE = r"C:\Users\16960\Desktop\期末论文"
SRC = BASE + r"\三模态数据库建立说明\scripts\Aeolus_V2\dataset\Flight_Tabular\2024"
OUT_YEAR = BASE + r"\新架构\data\2024数据"
OUT_MONTH = BASE + r"\新架构\data\4月数据"

STATIC_WINDOW = 60   # 静态边时间窗(分钟)
STATIC_K = 5         # 静态边每节点最多后向近邻
NET_WINDOW = 90      # 传播边时间窗(分钟)

RENAMES = {
    'DEP_DELAY': '出发延误', 'OP_CARRIER': '承运人', 'OP_CARRIER_FL_NUM': '航班号',
    'FL_YEAR': '年', 'FL_MONTH': '月', 'FL_DAY': '日', 'FL_WEEK': '星期',
    'ORIGIN_INDEX': '出发机场', 'DEST_INDEX': '到达机场',
    'CRS_DEP_TIME_MIN': '计划出发分钟', 'CRS_ARR_TIME_MIN': '计划到达分钟',
    'CRS_ELAPSED_TIME': '计划飞行时长', 'FLIGHTS': '航班计数',
    'O_TEMP': '出发地温度', 'O_PRCP': '出发地降水', 'O_WSPD': '出发地风速',
    'D_TEMP': '到达地温度', 'D_PRCP': '到达地降水', 'D_WSPD': '到达地风速',
    'O_LATITUDE': '出发地纬度', 'O_LONGITUDE': '出发地经度',
    'D_LATITUDE': '到达地纬度', 'D_LONGITUDE': '到达地经度', 'TAIL_NUM': '飞机尾号',
}


def load(period):
    if period == "month":
        files = sorted(glob.glob(os.path.join(SRC, "04", "*.csv")))
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        df = pd.read_csv(os.path.join(OUT_YEAR, "flight_2024.csv"))
    df.insert(0, '航班ID', range(len(df)))
    return df.rename(columns=RENAMES)


def build_chain(df):
    g = df.sort_values('计划出发分钟').groupby(
        ['飞机尾号', '年', '月', '日'], sort=False)
    nxt_id = g['航班ID'].shift(-1)
    nxt_dep = g['计划出发分钟'].shift(-1)
    cur_arr = df['计划到达分钟']
    mask = nxt_id.notna() & (nxt_dep - cur_arr).notna()
    out = pd.DataFrame({
        '前序航班ID': df.loc[mask, '航班ID'],
        '后续航班ID': nxt_id[mask].astype(int),
        '飞机尾号': df.loc[mask, '飞机尾号'],
        '周转间隔分钟': (nxt_dep - cur_arr)[mask],
    })
    return out.reset_index(drop=True)


def build_graph(df):
    edges = []
    # 静态边: 同机场同天, 后向 K 近邻且时间窗内, 再对称
    g = df.sort_values('计划出发分钟').groupby(
        ['出发机场', '年', '月', '日'], sort=False)
    dep = df['计划出发分钟']
    for s in range(1, STATIC_K + 1):
        nid = g['航班ID'].shift(-s)
        ndep = g['计划出发分钟'].shift(-s)
        m = nid.notna() & ((ndep - dep) <= STATIC_WINDOW)
        a = df.loc[m, '航班ID']; b = nid[m].astype(int); iv = (ndep - dep)[m]
        edges.append(pd.DataFrame({'源航班ID': a, '目标航班ID': b, '边类型': '静态', '间隔分钟': iv}))
        edges.append(pd.DataFrame({'源航班ID': b, '目标航班ID': a, '边类型': '静态', '间隔分钟': iv}))
    # 传播边: 到达某机场后窗口内下一个起飞
    dep_t = df[['出发机场', '计划出发分钟', '航班ID']].dropna().rename(
        columns={'出发机场': 'ap', '计划出发分钟': 'dep_time', '航班ID': 'dep_id'})
    arr_t = df[['到达机场', '计划到达分钟', '航班ID']].dropna().rename(
        columns={'到达机场': 'ap', '计划到达分钟': 'arr_time', '航班ID': 'arr_id'})
    mg = pd.merge_asof(arr_t.sort_values('arr_time'), dep_t.sort_values('dep_time'),
                       left_on='arr_time', right_on='dep_time', by='ap',
                       direction='forward', tolerance=NET_WINDOW)
    mg = mg.dropna(subset=['dep_id'])
    mg = mg[mg['dep_id'] != mg['arr_id']]
    edges.append(pd.DataFrame({
        '源航班ID': mg['arr_id'].astype(int), '目标航班ID': mg['dep_id'].astype(int),
        '边类型': '传播', '间隔分钟': mg['dep_time'] - mg['arr_time']}))
    return pd.concat(edges, ignore_index=True)


def main(period):
    out_dir = OUT_MONTH if period == "month" else OUT_YEAR
    tag = "4月" if period == "month" else "2024"
    os.makedirs(out_dir, exist_ok=True)

    df = load(period)
    print(f"[{period}] rows={len(df)}")

    tab = df[['航班ID'] + list(RENAMES.values())]
    tab.to_csv(os.path.join(out_dir, f"{tag}表格.csv"), index=False, encoding='utf-8-sig')
    print(f"[{period}] 表格 rows={len(tab)} cols={len(tab.columns)}")

    ch = build_chain(df)
    ch.to_csv(os.path.join(out_dir, f"{tag}链式.csv"), index=False, encoding='utf-8-sig')
    print(f"[{period}] 链式 edges={len(ch)}")

    gr = build_graph(df)
    gr.to_csv(os.path.join(out_dir, f"{tag}图.csv"), index=False, encoding='utf-8-sig')
    print(f"[{period}] 图 edges={len(gr)}")
    print(f"[{period}] DONE -> {out_dir}")


# ===== 可调变量 (直接改这里) =====
PERIOD = "month"   # month=4月 / year=全年
# ================================

if __name__ == "__main__":
    main(PERIOD)
