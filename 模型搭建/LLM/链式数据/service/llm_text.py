"""链上下文提示构建 (方案 B, 严格无泄露)。

为每个航班生成一段文本: 当前航班 + 当天同 (航司,航班号) 的前序航段 (按计划起飞排序)。
只用静态字段 (航司 / IATA 机场 / 计划时刻 / 天气 / 日期), 绝不含任何实际延误。
目的: 让冻结的 Gemma 调用世界知识 (枢纽拥堵、航司、季节、天气语义、当天累积),
产出 15 个数值特征之外的语义表征。
"""
import numpy as np

_WEEKDAY = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
_MONTH = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}

# 供 build_field_map 使用的列
_FIELD_COLS = [
    "sample_id", "OP_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN_INDEX", "DEST_INDEX",
    "FL_DATE", "FL_MONTH", "FL_WEEK", "CRS_DEP_TIME_MIN", "CRS_ARR_TIME_MIN",
    "O_TEMP", "D_TEMP", "O_PRCP", "D_PRCP", "O_WSPD", "D_WSPD",
]


def _hhmm(minutes):
    try:
        m = int(minutes)
    except (TypeError, ValueError):
        return "NA"
    if m < 0:
        return "NA"
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


def _num(value, digits=0):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(f):
        return "NA"
    return f"{f:.{digits}f}"


def build_field_map(df, needed_ids=None):
    """df -> {sample_id(str): 字段 dict}。needed_ids 给定时只保留这些航班。"""
    sub = df
    if needed_ids is not None:
        needed = set(map(str, needed_ids))
        sub = df[df["sample_id"].astype(str).isin(needed)]
    cols = [c for c in _FIELD_COLS if c in sub.columns]
    sub = sub[cols]
    field_map = {}
    for row in sub.itertuples(index=False):
        d = row._asdict()
        sid = str(d["sample_id"])
        field_map[sid] = d
    return field_map


def _weather(temp, prcp, wspd):
    return f"temp {_num(temp)}C, precip {_num(prcp, 2)}, wind {_num(wspd)}"


def render_leg(f, ordinal=None):
    """一段航段的简短描述 (静态)。"""
    tag = f"leg {ordinal}: " if ordinal is not None else ""
    return (
        f"{tag}{f['ORIGIN_INDEX']}->{f['DEST_INDEX']} "
        f"sched {_hhmm(f['CRS_DEP_TIME_MIN'])}->{_hhmm(f['CRS_ARR_TIME_MIN'])}, "
        f"origin wx {_weather(f['O_TEMP'], f['O_PRCP'], f['O_WSPD'])}"
    )


def render_prompt(cur, prev_list):
    """当前航班 + 前序航段 -> 完整提示 (静态)。"""
    try:
        wd = _WEEKDAY.get(int(cur["FL_WEEK"]), str(cur["FL_WEEK"]))
    except (TypeError, ValueError):
        wd = str(cur.get("FL_WEEK"))
    try:
        mo = _MONTH.get(int(cur["FL_MONTH"]), str(cur["FL_MONTH"]))
    except (TypeError, ValueError):
        mo = str(cur.get("FL_MONTH"))

    lines = [
        "Task: Will this US flight depart more than 15 minutes late? "
        "Consider airport congestion, carrier, weather, time of day, and delays accumulated from earlier legs.",
        f"Flight: carrier {cur['OP_CARRIER']}, number {cur['OP_CARRIER_FL_NUM']}, "
        f"route {cur['ORIGIN_INDEX']}->{cur['DEST_INDEX']}. "
        f"Date {cur['FL_DATE']} ({wd}, {mo}). "
        f"Scheduled {_hhmm(cur['CRS_DEP_TIME_MIN'])} depart, {_hhmm(cur['CRS_ARR_TIME_MIN'])} arrive. "
        f"Origin weather {_weather(cur['O_TEMP'], cur['O_PRCP'], cur['O_WSPD'])}. "
        f"Dest weather {_weather(cur['D_TEMP'], cur['D_PRCP'], cur['D_WSPD'])}.",
    ]
    if prev_list:
        lines.append("Earlier legs today (same flight number), in scheduled order:")
        for i, pf in enumerate(prev_list, start=1):
            lines.append("  " + render_leg(pf, ordinal=i))
        lines.append(f"This is leg {len(prev_list) + 1} of the day.")
    else:
        lines.append("This is the first scheduled leg of the day.")
    return "\n".join(lines)


def build_texts(sid_out, valid_len, field_map, max_prev):
    """按链张量的排序位置生成 {sample_id: prompt}, 与张量位置严格对齐。"""
    texts = {}
    sid_out = np.asarray(sid_out, dtype=object)
    valid_len = np.asarray(valid_len)
    for chain_idx in range(sid_out.shape[0]):
        v = int(valid_len[chain_idx])
        legs = [str(sid_out[chain_idx, t]) for t in range(v)]
        for t in range(v):
            cur_id = legs[t]
            cur = field_map.get(cur_id)
            if cur is None:
                continue
            start = max(0, t - max_prev)
            prev_list = [field_map[legs[j]] for j in range(start, t) if legs[j] in field_map]
            texts[cur_id] = render_prompt(cur, prev_list)
    return texts
