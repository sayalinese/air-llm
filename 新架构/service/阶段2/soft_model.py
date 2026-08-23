# -*- coding: utf-8 -*-
"""真软提示注入模型: GAT向量 -> K个软token 注入 Gemma4 文本层 + 分类头。无门控融合。"""
import torch
from torch import nn

import config2 as C


class SoftPromptNet(nn.Module):
    """投影 GAT 向量为 K 个软token, 拼进 LLM 输入; 头读 LLM 隐藏态分类。"""

    def __init__(self, H, z_dim=128):
        super().__init__()
        self.proj = nn.Linear(z_dim, C.NUM_SOFT * H)
        self.head = nn.Linear(H, 1)

    def forward(self, txt, z, ids, mask, pad_id):
        B, H = ids.size(0), self.head.in_features
        K = C.NUM_SOFT
        dummy = ids.new_full((B, K), pad_id)
        full_ids = torch.cat([dummy, ids], 1)
        with torch.no_grad():
            pli = txt.get_per_layer_inputs(full_ids, None)   # 预计算, 防词表反向爆显存
        te = txt.embed_tokens(ids)
        soft = self.proj(z).view(B, K, H)
        ie = torch.cat([soft, te], 1)
        am = torch.cat([torch.ones(B, K, device=ie.device, dtype=mask.dtype), mask], 1)
        out = txt(inputs_embeds=ie, per_layer_inputs=pli, attention_mask=am)
        h = out.last_hidden_state[:, :K].mean(1)             # 软token 经LLM后的隐藏态
        return self.head(h).squeeze(-1)
