# -*- coding: utf-8 -*-
"""真软提示注入模型: GAT向量 -> K个软token 注入 Gemma4 文本层 + 分类头。无门控融合。
消融开关 (config2): PROJ_NORM(E1) / POOL_ATTN(E3) / HEAD_MLP(E4), NUM_SOFT(E2), LORA_R(ALPHA)(E5)。"""
import torch
from torch import nn

import config2 as C


class SoftPromptNet(nn.Module):
    """投影 GAT 向量为 K 个软token, 拼进 LLM 输入; 头读 LLM 隐藏态分类。"""

    def __init__(self, H, z_dim=128, readout_mode=None):
        super().__init__()
        self.readout_mode = readout_mode or C.READOUT_MODE
        if self.readout_mode not in {"soft_prefix", "text_then_soft", "cls"}:
            raise ValueError(f"Unsupported READOUT_MODE={self.readout_mode}")
        self.H = H
        layers = []
        if C.PROJ_NORM:                       # E1: z 输入归一化
            layers.append(nn.LayerNorm(z_dim))
        layers.append(nn.Linear(z_dim, C.NUM_SOFT * H))
        if C.PROJ_NORM:                       # E1: 投影后非线性 (与归一绑定)
            layers.append(nn.GELU())
        self.proj = nn.Sequential(*layers)
        if self.readout_mode == "cls":
            self.cls_token = nn.Parameter(torch.empty(1, 1, H))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        if C.TOKEN_GATE:                      # S1: 读出门控, 偏置+2初始g≈0.88
            self.gate_logits = nn.Parameter(torch.full((C.NUM_SOFT,), 2.0))
        if C.POOL_ATTN:                       # E3: 注意力池化 (学习 query 加权求和)
            self.pool_q = nn.Parameter(torch.zeros(1, 1, H))
            nn.init.xavier_uniform_(self.pool_q)
        in_dim = H + (z_dim if C.Z_SHORTCUT else 0)  # N2: z旁路直连head
        if C.Z_SHORTCUT:
            self.z_norm = nn.LayerNorm(z_dim)
        if C.HEAD_MLP:                        # E4: 两层 MLP 分类头
            self.head = nn.Sequential(
                nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim // 4), nn.GELU(), nn.Linear(in_dim // 4, 1))
        else:
            self.head = nn.Linear(in_dim, 1)

    def forward(self, txt, z, ids, mask, pad_id):
        B, H, K = ids.size(0), self.H, C.NUM_SOFT
        dummy = ids.new_full((B, K), pad_id)
        te = txt.embed_tokens(ids)
        soft = self.proj(z).view(B, K, H)
        soft_mask = torch.ones(B, K, device=soft.device, dtype=mask.dtype)

        if self.readout_mode == "soft_prefix":
            full_ids = torch.cat([dummy, ids], 1)
            ie = torch.cat([soft, te], 1)
            am = torch.cat([soft_mask, mask], 1)
            soft_slice = slice(0, K)
        elif self.readout_mode == "text_then_soft":
            full_ids = torch.cat([ids, dummy], 1)
            ie = torch.cat([te, soft], 1)
            am = torch.cat([mask, soft_mask], 1)
            soft_slice = slice(-K, None)
        else:
            cls_dummy = ids.new_full((B, 1), pad_id)
            cls = self.cls_token.expand(B, -1, -1)
            cls_mask = torch.ones(B, 1, device=soft.device, dtype=mask.dtype)
            full_ids = torch.cat([dummy, ids, cls_dummy], 1)
            ie = torch.cat([soft, te, cls], 1)
            am = torch.cat([soft_mask, mask, cls_mask], 1)
            soft_slice = slice(0, K)

        with torch.no_grad():
            pli = txt.get_per_layer_inputs(full_ids, None)   # 预计算, 防词表反向爆显存
        out = txt(inputs_embeds=ie, per_layer_inputs=pli, attention_mask=am)
        h_soft = out.last_hidden_state[:, soft_slice]
        if self.readout_mode == "cls":
            h = out.last_hidden_state[:, -1]
        elif C.TOKEN_GATE:
            g = torch.sigmoid(self.gate_logits.float()).view(1, K, 1)   # fp32防bf16吞更新
            h = ((h_soft.float() * g).sum(1) / (g.sum() + 1e-6)).to(h_soft.dtype)  # S1: 读出门控
        elif C.POOL_ATTN:
            w = torch.softmax((h_soft * self.pool_q).sum(-1), dim=-1).unsqueeze(-1)
            h = (h_soft * w).sum(1)
        else:
            h = h_soft.mean(1)
        if C.Z_SHORTCUT:                      # N2: 原始z旁路进head, 下限锁定为线性头读z
            h = torch.cat([h, self.z_norm(z)], 1)
        return self.head(h).squeeze(-1)
