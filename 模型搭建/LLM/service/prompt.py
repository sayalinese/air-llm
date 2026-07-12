"""Canonical compact prompt rendering shared by preprocessing and model code."""


STAGE_ORDER = {"current": 0, "chain": 1, "context": 2, "nearby": 3, "risk": 4}


def _fmt(value, digits=2):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "1" if value else "0"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _rate(summary):
    value = summary.get("delay_rate")
    return 0.0 if value is None else max(0.0, min(1.0, float(value)))


def _risk_line(tail, route, reverse, airport, nearby):
    prev_arr = tail.get("prev_arr_delay") if tail.get("prev_arrived") else None
    slack = tail.get("turnaround_slack_min")
    chain_risk = 0.0
    if prev_arr is not None and slack is not None:
        chain_risk = max(0.0, min(1.0, max(0.0, float(prev_arr)) / max(float(slack), 1.0)))
    airport_risk = (
        0.50 * _rate(airport["origin_60m"])
        + 0.25 * _rate(airport["carrier_origin_60m"])
        + 0.15 * _rate(airport["dest_60m"])
        + 0.10 * _rate(nearby["near_origin_60m"])
    )
    route_risk = 0.75 * _rate(route) + 0.25 * _rate(reverse)
    total = 0.50 * chain_risk + 0.30 * airport_risk + 0.20 * route_risk
    return f"综合险:{int(round(total * 100))}"


def render_compact_features(features, chain_position=None, chain_valid_length=None, stage="nearby"):
    """Render only the feature groups used by the LLM classifier."""
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown ablation stage: {stage}")
    cur = features["current"]
    tail = features["tail_chain"]
    route = features["route_context"]
    reverse = features["reverse_route_context"]
    airport = features["airport_context"]
    nearby = features["nearby_airport_context"]

    origin_weather = cur.get("origin_weather") or {}
    position = chain_position if chain_position is not None else "NA"
    length = chain_valid_length if chain_valid_length is not None else "NA"

    lines = [
            (
                f"当前: 月{_fmt(cur.get('month'))} 周{_fmt(cur.get('day_of_week'))} "
                f"{cur.get('dep_time', 'NA')} {cur.get('origin', 'NA')}>{cur.get('dest', 'NA')} "
                f"承{cur.get('carrier', 'NA')} 飞{_fmt(cur.get('elapsed_min'))} 段{position}/{length} "
                f"cutoff={cur.get('observation_time', 'NA')} T-{_fmt(cur.get('prediction_horizon_min'))}"
            ),
        ]
    if any(origin_weather.get(key) is not None for key in ("temp", "prcp", "wspd")):
        lines.append(
            f"天气: 起温{_fmt(origin_weather.get('temp'))} 雨{_fmt(origin_weather.get('prcp'))} "
            f"风{_fmt(origin_weather.get('wspd'))}"
        )
    if STAGE_ORDER[stage] >= STAGE_ORDER["chain"]:
        lines.append(
            (
                f"同机: 有前序{_fmt(tail.get('has_previous'))} "
                f"已起飞{_fmt(tail.get('prev_departed'))} 已到达{_fmt(tail.get('prev_arrived'))} "
                f"出延{_fmt(tail.get('prev_dep_delay'))} 到延{_fmt(tail.get('prev_arr_delay'))} "
                f"缓冲{_fmt(tail.get('turnaround_slack_min'))} 趋势{_fmt(tail.get('dep_delay_trend'))}"
            )
        )
    if STAGE_ORDER[stage] >= STAGE_ORDER["context"]:
        lines.extend([
            (
                f"机场60: 起n{_fmt(airport['origin_60m'].get('count'))} "
                f"率{_fmt(airport['origin_60m'].get('delay_rate'))}; "
                f"承运起n{_fmt(airport['carrier_origin_60m'].get('count'))} "
                f"率{_fmt(airport['carrier_origin_60m'].get('delay_rate'))}; "
                f"到n{_fmt(airport['dest_60m'].get('count'))} "
                f"率{_fmt(airport['dest_60m'].get('delay_rate'))}"
            ),
            (
                f"航段5: n{_fmt(route.get('count'))} 率{_fmt(route.get('delay_rate'))}; "
                f"反向n{_fmt(reverse.get('count'))} 率{_fmt(reverse.get('delay_rate'))}"
            ),
        ])
    if STAGE_ORDER[stage] >= STAGE_ORDER["nearby"]:
        lines.append(
            (
                f"邻: 起{_fmt(nearby['near_origin_60m'].get('count'))}/"
                f"{_fmt(nearby['near_origin_60m'].get('delay_rate'))} "
                f"到{_fmt(nearby['near_dest_60m'].get('count'))}/"
                f"{_fmt(nearby['near_dest_60m'].get('delay_rate'))}"
            )
        )
    if STAGE_ORDER[stage] >= STAGE_ORDER["risk"]:
        lines.append(_risk_line(tail, route, reverse, airport, nearby))
    return "\n".join(lines)
