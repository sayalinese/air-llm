# -*- coding: utf-8 -*-
"""离线历史检索工具，避免训练时逐批访问外部向量数据库。"""
from collections import defaultdict

import torch
from torch.nn import functional as F


def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _key(meta, fields):
    return tuple(meta[field] for field in fields)


def build_history_index(bank_meta):
    """建立按日期分桶的结构索引，索引值均为训练库行号。"""
    fields = {
        "carrier_route": ("carrier", "origin", "destination"),
        "route": ("origin", "destination"),
        "carrier_origin": ("carrier", "origin"),
        "carrier_destination": ("carrier", "destination"),
    }
    indices = {name: defaultdict(lambda: defaultdict(list)) for name in fields}
    by_day = defaultdict(list)
    for row, meta in enumerate(bank_meta):
        day = int(meta["day"])
        by_day[day].append(row)
        for name, names in fields.items():
            indices[name][_key(meta, names)][day].append(row)

    prefix = {}
    previous = []
    for day in range(1, max(by_day.keys(), default=0) + 2):
        prefix[day] = list(previous)
        previous.extend(by_day.get(day, []))
    return indices, prefix


def _prior_for(mapping, key, day):
    values = mapping.get(key)
    if not values:
        return []
    out = []
    for candidate_day, rows in values.items():
        if candidate_day < day:
            out.extend(rows)
    return out


def _structural_candidates(meta, indices, prefix, day, cap):
    """按层级回退收集候选；同一天及之后的样本永远不会进入候选。"""
    tiers = [
        (1, "carrier_route", (meta["carrier"], meta["origin"], meta["destination"])),
        (2, "route", (meta["origin"], meta["destination"])),
        (3, "carrier_origin", (meta["carrier"], meta["origin"])),
        (3, "carrier_destination", (meta["carrier"], meta["destination"])),
    ]
    selected, levels = [], {}
    for level, name, key in tiers:
        for row in _prior_for(indices[name], key, day):
            if row not in levels:
                levels[row] = level
                selected.append(row)
                if len(selected) >= cap:
                    return selected, levels
    return selected, levels


def _weather_distance(query, bank):
    return (abs(query["temperature"] - bank["temperature"]) / 20.0
            + abs(query["wind"] - bank["wind"]) / 20.0
            + abs(query["precipitation"] - bank["precipitation"]) / 5.0)


@torch.no_grad()
def hybrid_topk(query_z, bank_z, query_meta, bank_meta, k,
                query_labels=None, bank_labels=None, strategy="hybrid",
                query_batch_size=512, vector_recall_k=128,
                struct_candidate_cap=256, device="cuda", config=None,
                seed=42):
    """严格历史候选上的向量/随机/混合检索。

    返回固定形状的索引、相似度、层级、有效掩码和候选数。
    无可用历史时使用 -1 填充，不会伪造案例。
    """
    if strategy not in {"hybrid", "vector", "random"}:
        raise ValueError(f"Unsupported retrieval strategy={strategy}")
    if len(query_z) != len(query_meta) or len(bank_z) != len(bank_meta):
        raise ValueError("metadata and z row counts must match")
    if k <= 0:
        raise ValueError("k must be positive")

    class W:
        ROUTE_BONUS = 0.08; CARRIER_BONUS = 0.03; ORIGIN_BONUS = 0.02
        DESTINATION_BONUS = 0.02; WEEKDAY_BONUS = 0.015; TIME_BONUS = 0.04
        WEATHER_BONUS = 0.02; TIME_SCALE_MINUTES = 240.0; WEATHER_SCALE = 3.0
    w = config or W
    dev = resolve_device(device)
    qz = F.normalize(query_z.float().cpu(), dim=1)
    bz = F.normalize(bank_z.float().cpu(), dim=1)
    bank_day = torch.tensor([int(m["day"]) for m in bank_meta], dtype=torch.long)
    indices, prefix = build_history_index(bank_meta)
    max_query_day = max([int(m["day"]) for m in query_meta], default=0)
    max_bank_day = max([int(m["day"]) for m in bank_meta], default=0)
    if max_query_day > max_bank_day + 1:
        previous = []
        by_day = defaultdict(list)
        for row, meta in enumerate(bank_meta):
            by_day[int(meta["day"])].append(row)
        for day in range(1, max_query_day + 1):
            prefix[day] = list(previous)
            previous.extend(by_day.get(day, []))
    bank_tensors = {
        name: torch.tensor([m[name] for m in bank_meta], dtype=torch.float32)
        for name in ("temperature", "wind", "precipitation")
    }
    bz_dev = bz.to(dev)
    bank_day_dev = bank_day.to(dev)
    bank_carrier_dev = torch.tensor([int(m["carrier_code"]) for m in bank_meta], device=dev)
    bank_origin_dev = torch.tensor([int(m["origin_code"]) for m in bank_meta], device=dev)
    bank_destination_dev = torch.tensor([int(m["destination_code"]) for m in bank_meta], device=dev)
    bank_weekday_dev = torch.tensor([int(m["weekday"]) for m in bank_meta], device=dev)
    bank_departure_dev = torch.tensor([float(m["departure_minute"]) for m in bank_meta], device=dev)
    bank_temperature_dev = bank_tensors["temperature"].to(dev)
    bank_wind_dev = bank_tensors["wind"].to(dev)
    bank_precipitation_dev = bank_tensors["precipitation"].to(dev)

    out_idx, out_score, out_level, out_valid, out_count = [], [], [], [], []
    for q0 in range(0, len(query_z), query_batch_size):
        q1 = min(q0 + query_batch_size, len(query_z))
        q_batch = qz[q0:q1].to(dev)
        q_days = torch.tensor([int(m["day"]) for m in query_meta[q0:q1]], dtype=torch.long)
        eligible = q_days.to(dev)[:, None] > bank_day_dev[None, :]
        if strategy != "random":
            vec_k = min(vector_recall_k, len(bank_meta))
            vec_values = torch.full((q1 - q0, vec_k), float("-inf"), device=dev)
            vec_rows = torch.full((q1 - q0, vec_k), -1, dtype=torch.long, device=dev)
            chunk_size = max(1, int(getattr(config, "BANK_CHUNK_SIZE", 40000)))
            for b0 in range(0, len(bank_meta), chunk_size):
                b1 = min(b0 + chunk_size, len(bank_meta))
                local_scores = q_batch @ bz_dev[b0:b1].T
                local_scores.masked_fill_(~eligible[:, b0:b1], float("-inf"))
                local_k = min(vec_k, b1 - b0)
                local_values, local_rows = local_scores.topk(local_k, dim=1)
                local_rows += b0
                merged_values = torch.cat([vec_values, local_values], dim=1)
                merged_rows = torch.cat([vec_rows, local_rows], dim=1)
                vec_values, positions = merged_values.topk(vec_k, dim=1)
                vec_rows = merged_rows.gather(1, positions)
        batch_lists, batch_levels, batch_counts = [], [], []

        for local, meta in enumerate(query_meta[q0:q1]):
            day = int(meta["day"])
            prior_rows = prefix.get(day, [])
            if strategy == "random":
                generator = torch.Generator().manual_seed(seed + int(meta["sample_id"]))
                if len(prior_rows) > 0:
                    perm = torch.randperm(len(prior_rows), generator=generator).tolist()
                    rows = [prior_rows[i] for i in perm[:k]]
                else:
                    rows = []
                levels = {row: 4 for row in rows}
            elif strategy == "vector":
                rows = [int(row) for row, score in zip(
                    vec_rows[local].cpu().tolist(), vec_values[local].cpu().tolist())
                    if score != float("-inf")]
                levels = {row: 4 for row in rows}
            else:
                rows, levels = _structural_candidates(
                    meta, indices, prefix, day, struct_candidate_cap)
                for row, score in zip(vec_rows[local].cpu().tolist(), vec_values[local].cpu().tolist()):
                    row = int(row)
                    if score != float("-inf") and row not in levels:
                        levels[row] = 4
                        rows.append(row)

            rows = rows[:max(k, struct_candidate_cap if strategy == "hybrid" else vector_recall_k)]
            batch_lists.append(rows)
            batch_levels.append(levels)
            batch_counts.append(len(rows))

        max_c = max([len(x) for x in batch_lists], default=0)
        if max_c == 0:
            out_idx.append(torch.full((q1 - q0, k), -1, dtype=torch.long))
            out_score.append(torch.full((q1 - q0, k, 2), float("-inf")))
            out_level.append(torch.full((q1 - q0, k), -1, dtype=torch.long))
            out_valid.append(torch.zeros((q1 - q0, k), dtype=torch.bool))
            out_count.append(torch.tensor(batch_counts, dtype=torch.long))
            continue

        cand = torch.full((q1 - q0, max_c), -1, dtype=torch.long)
        for row, rows in enumerate(batch_lists):
            if rows:
                cand[row, :len(rows)] = torch.tensor(rows, dtype=torch.long)
        valid = cand >= 0
        safe_cand = cand.clamp_min(0)
        cand_z = bz[safe_cand].to(dev)
        vector_scores = (q_batch[:, None, :] * cand_z).sum(-1)
        vector_scores.masked_fill_(~valid.to(dev), float("-inf"))
        scores = vector_scores.clone()

        if strategy == "hybrid":
            qmeta = query_meta[q0:q1]
            carrier = torch.tensor([int(m["carrier_code"]) for m in qmeta], device=dev)[:, None]
            origin = torch.tensor([int(m["origin_code"]) for m in qmeta], device=dev)[:, None]
            destination = torch.tensor([int(m["destination_code"]) for m in qmeta], device=dev)[:, None]
            weekday = torch.tensor([int(m["weekday"]) for m in qmeta], device=dev)[:, None]
            dep = torch.tensor([float(m["departure_minute"]) for m in qmeta], device=dev)[:, None]
            bcarrier = bank_carrier_dev[safe_cand]
            borigin = bank_origin_dev[safe_cand]
            bdest = bank_destination_dev[safe_cand]
            bweekday = bank_weekday_dev[safe_cand]
            bdep = bank_departure_dev[safe_cand]
            btemp = bank_temperature_dev[safe_cand]
            bwind = bank_wind_dev[safe_cand]
            bprecip = bank_precipitation_dev[safe_cand]
            qtemp = torch.tensor([float(m["temperature"]) for m in qmeta], device=dev)[:, None]
            qwind = torch.tensor([float(m["wind"]) for m in qmeta], device=dev)[:, None]
            qprecip = torch.tensor([float(m["precipitation"]) for m in qmeta], device=dev)[:, None]
            route = (origin == borigin) & (destination == bdest)
            scores += route.float() * w.ROUTE_BONUS
            scores += (carrier == bcarrier).float() * w.CARRIER_BONUS
            scores += (origin == borigin).float() * w.ORIGIN_BONUS
            scores += (destination == bdest).float() * w.DESTINATION_BONUS
            scores += (weekday == bweekday).float() * w.WEEKDAY_BONUS
            scores += torch.exp(-torch.abs(dep - bdep) / w.TIME_SCALE_MINUTES) * w.TIME_BONUS
            weather = (torch.abs(qtemp - btemp) / 20.0
                       + torch.abs(qwind - bwind) / 20.0
                       + torch.abs(qprecip - bprecip) / 5.0)
            scores += torch.exp(-weather / w.WEATHER_SCALE) * w.WEATHER_BONUS

        take = min(k, max_c)
        values, positions = scores.topk(take, dim=1)
        selected_vector_scores = vector_scores.gather(1, positions).cpu()
        chosen = cand.to(dev).gather(1, positions).cpu()
        chosen_levels = torch.full_like(chosen, 4)
        for row, levels in enumerate(batch_levels):
            for col, bank_row in enumerate(chosen[row].tolist()):
                chosen_levels[row, col] = int(levels.get(int(bank_row), 4)) if bank_row >= 0 else -1
        chosen_valid = torch.isfinite(values).cpu()
        if take < k:
            pad = k - take
            chosen = torch.cat([chosen, torch.full((q1 - q0, pad), -1, dtype=torch.long)], 1)
            values = torch.cat([values.cpu(), torch.full((q1 - q0, pad), float("-inf"))], 1)
            selected_vector_scores = torch.cat(
                [selected_vector_scores, torch.full((q1 - q0, pad), float("-inf"))], 1)
            chosen_levels = torch.cat([chosen_levels, torch.full((q1 - q0, pad), -1, dtype=torch.long)], 1)
            chosen_valid = torch.cat([chosen_valid, torch.zeros((q1 - q0, pad), dtype=torch.bool)], 1)
        out_idx.append(chosen)
        out_score.append(torch.stack([selected_vector_scores, values.cpu()], dim=-1))
        out_level.append(chosen_levels)
        out_valid.append(chosen_valid)
        out_count.append(torch.tensor(batch_counts, dtype=torch.long))

    scores = torch.cat(out_score)
    return (torch.cat(out_idx), scores[..., 0], scores[..., 1],
            torch.cat(out_level), torch.cat(out_valid), torch.cat(out_count))


@torch.no_grad()
def cosine_topk(query_z, bank_z, k, query_ids=None, bank_ids=None,
                exclude_self=False, query_batch_size=512,
                bank_chunk_size=40000, device="cuda"):
    """分块计算余弦top-k，返回检索库行号和相似度。

    train->train 检索必须传入ID并设置 exclude_self=True，防止样本检索到自身。
    """
    if query_z.ndim != 2 or bank_z.ndim != 2:
        raise ValueError("query_z and bank_z must be rank-2 tensors")
    if query_z.shape[1] != bank_z.shape[1]:
        raise ValueError("query_z and bank_z must have the same feature dimension")
    if k <= 0 or k >= len(bank_z) + int(not exclude_self):
        raise ValueError(f"Invalid k={k} for bank size {len(bank_z)}")
    if exclude_self and (query_ids is None or bank_ids is None):
        raise ValueError("exclude_self=True requires query_ids and bank_ids")

    dev = resolve_device(device)
    bank_z = F.normalize(bank_z.float(), dim=1).cpu()
    query_z = query_z.float().cpu()
    query_ids = torch.as_tensor(query_ids, dtype=torch.long) if query_ids is not None else None
    bank_ids = torch.as_tensor(bank_ids, dtype=torch.long) if bank_ids is not None else None
    all_indices, all_scores = [], []

    for q0 in range(0, len(query_z), query_batch_size):
        q1 = min(q0 + query_batch_size, len(query_z))
        q = F.normalize(query_z[q0:q1], dim=1).to(dev)
        best_scores = torch.full((len(q), k), float("-inf"), device=dev)
        best_indices = torch.full((len(q), k), -1, dtype=torch.long, device=dev)

        for b0 in range(0, len(bank_z), bank_chunk_size):
            b1 = min(b0 + bank_chunk_size, len(bank_z))
            scores = q @ bank_z[b0:b1].to(dev).T
            if exclude_self:
                same_id = query_ids[q0:q1].to(dev)[:, None] == bank_ids[b0:b1].to(dev)[None, :]
                scores.masked_fill_(same_id, float("-inf"))

            local_k = min(k, b1 - b0)
            local_scores, local_indices = scores.topk(local_k, dim=1)
            local_indices += b0
            merged_scores = torch.cat([best_scores, local_scores], dim=1)
            merged_indices = torch.cat([best_indices, local_indices], dim=1)
            best_scores, positions = merged_scores.topk(k, dim=1)
            best_indices = merged_indices.gather(1, positions)

        if (best_indices < 0).any() or not torch.isfinite(best_scores).all():
            raise RuntimeError("Unable to produce complete top-k retrieval results")
        all_indices.append(best_indices.cpu())
        all_scores.append(best_scores.float().cpu())

    return torch.cat(all_indices), torch.cat(all_scores)
