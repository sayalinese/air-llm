"""真序列: 为每个目标航班构造"起飞机场近窗口计划航班序列"提示 (严格合规)。

合规口径:
- 邻居 = 与目标航班"同机场(ORIGIN_INDEX)、同日(FL_DATE)、计划起飞早于目标 T 且在 T-窗口 内"的航班;
- 只用静态/计划字段 (航司 / 目的地 / 计划时刻 / 天气); 绝不含实际延误、实际起降时刻;
- 只取 T 之前的航班 (无未来泄露); 目标自身不计入。
产出一段真 token 序列 (K 条邻居 + 目标航班), 让冻结 LLM 的注意力建模"机场拥堵态势"。
"""
import numpy as np  # noqa: F401 (预留)

from .llm_text import _hhmm, _MONTH, _WEEKDAY, _weather

_COLS = [
    "sample_id", "OP_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN_INDEX", "DEST_INDEX",
    "FL_DATE", "FL_MONTH", "FL_WEEK", "CRS_DEP_TIME_MIN", "CRS_ARR_TIME_MIN",
    "O_TEMP", "D_TEMP", "O_PRCP", "D_PRCP", "O_WSPD", "D_WSPD",
]


def _dep_min(r):
    try:
        return int(float(r["CRS_DEP_TIME_MIN"]))
    except (TypeError, ValueError):
        return -1


def _neighbor_line(r):
    return (f"  - {r['OP_CARRIER']} to {r['DEST_INDEX']} at {_hhmm(r['CRS_DEP_TIME_MIN'])}, "
            f"wx {_weather(r['O_TEMP'], r['O_PRCP'], r['O_WSPD'])}")


def render_airseq_prompt(cur, neighbors, window_min):
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
        "Judge airport congestion from the recent scheduled departures listed below.",
        f"Flight: carrier {cur['OP_CARRIER']}, number {cur['OP_CARRIER_FL_NUM']}, "
        f"route {cur['ORIGIN_INDEX']}->{cur['DEST_INDEX']}. Date {cur['FL_DATE']} ({wd}, {mo}). "
        f"Scheduled {_hhmm(cur['CRS_DEP_TIME_MIN'])} depart, {_hhmm(cur['CRS_ARR_TIME_MIN'])} arrive. "
        f"Origin weather {_weather(cur['O_TEMP'], cur['O_PRCP'], cur['O_WSPD'])}. "
        f"Dest weather {_weather(cur['D_TEMP'], cur['D_PRCP'], cur['D_WSPD'])}.",
    ]
    if neighbors:
        lines.append(f"Scheduled departures from {cur['ORIGIN_INDEX']} in the {window_min} min before "
                     f"(airport load, {len(neighbors)} flights):")
        lines.extend(_neighbor_line(r) for r in neighbors)
    else:
        lines.append(f"No other scheduled departures from {cur['ORIGIN_INDEX']} in the prior {window_min} min.")
    return "\n".join(lines)


def build_airseq_texts(df, needed_ids, window_min, max_neighbors):
    """{sample_id: prompt}; \u90bb\u5c45\u4ece\u5168\u91cf\u540c(\u65e5,\u673a\u573a)\u822a\u73ed\u4e2d\u53d6, \u4ec5\u524d\u5e8f\u3001\u9759\u6001\u3002"""
    needed = set(map(str, needed_ids))
    cols = [c for c in _COLS if c in df.columns]
    texts = {}
    for _, g in df[cols].groupby(["FL_DATE", "ORIGIN_INDEX"], sort=False):
        g_sids = set(g["sample_id"].astype(str))
        if not (g_sids & needed):
            continue
        recs = g.sort_values("CRS_DEP_TIME_MIN").to_dict("records")
        tvals = [_dep_min(r) for r in recs]
        for i, r in enumerate(recs):
            sid = str(r["sample_id"])
            if sid not in needed:
                continue
            t = tvals[i]
            nb = []
            for j in range(i - 1, -1, -1):
                if 0 <= tvals[j] < t and t - tvals[j] <= window_min:
                    nb.append(recs[j])
                    if len(nb) >= max_neighbors:
                        break
            texts[sid] = render_airseq_prompt(r, list(reversed(nb)), window_min)
    return texts
