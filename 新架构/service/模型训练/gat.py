# -*- coding: utf-8 -*-
"""三模态 GAT 模型结构 (纯 PyTorch, 零图库依赖)。

节点: 8 sparse Embedding + 7 dense -> 输入投影;
每图一个 GAT 分支 (多头图注意力); 融合: 三路表征共享注意力加权;
逐节点输出 logit (延误二分类)。
edge_index [2,E]: 行0=src, 行1=dst, 消息 src->dst 在 dst 聚合。
"""
import torch
import torch.nn.functional as F
from torch import nn


def _scatter_softmax(scores, index, num_nodes):
    """按 dst(index) 分组做 softmax -> 归一化注意力权重 [E,H]。"""
    max_per = scores.new_full((num_nodes, scores.size(1)), float("-inf"))
    max_per = max_per.index_reduce(0, index, scores, "amax", include_self=True)
    max_per = torch.where(torch.isinf(max_per), torch.zeros_like(max_per), max_per)
    exp = torch.exp(scores - max_per[index])
    denom = torch.zeros(num_nodes, scores.size(1), device=scores.device, dtype=scores.dtype)
    denom = denom.index_add(0, index, exp)
    return exp / (denom[index] + 1e-16)


class GATLayer(nn.Module):
    """单层多头图注意力: dst 汇聚 src 消息。"""

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        assert out_dim % heads == 0
        self.heads, self.dh = heads, out_dim // heads
        self.lin = nn.Linear(in_dim, out_dim)
        self.attn_src = nn.Parameter(torch.zeros(heads, self.dh))
        self.attn_dst = nn.Parameter(torch.zeros(heads, self.dh))
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x, edge_index):
        N = x.size(0)
        h = self.lin(x).view(N, self.heads, self.dh)
        if edge_index.size(1) == 0:
            return h.reshape(N, -1)
        src, dst = edge_index[0], edge_index[1]
        e = (h[src] * self.attn_src).sum(-1) + (h[dst] * self.attn_dst).sum(-1)
        e = F.leaky_relu(e, 0.2)
        alpha = self.drop(_scatter_softmax(e, dst, N))
        msg = h[src] * alpha.unsqueeze(-1)
        out = torch.zeros(N, self.heads, self.dh, device=x.device, dtype=x.dtype)
        out = out.index_add(0, dst, msg)
        return out.reshape(N, -1)


class NodeEncoder(nn.Module):
    """8 sparse Embedding + 7 dense -> hidden 输入表征 (三图共享)。"""

    def __init__(self, vocab_sizes, dense_dim, emb_dim, hidden, dropout=0.1):
        super().__init__()
        self.embeds = nn.ModuleList(
            [nn.Embedding(int(v), emb_dim, padding_idx=0) for v in vocab_sizes])
        in_dim = emb_dim * len(vocab_sizes) + dense_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))

    def forward(self, dense, sparse):
        embs = [emb(sparse[:, i]) for i, emb in enumerate(self.embeds)]
        return self.proj(torch.cat(embs + [dense], dim=-1))


class GATBranch(nn.Module):
    """单模态分支: 2 层 GAT + 残差 + LayerNorm。"""

    def __init__(self, hidden, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([GATLayer(hidden, hidden, heads, dropout) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index):
        for gat, norm in zip(self.layers, self.norms):
            h = norm(h + self.drop(F.gelu(gat(h, edge_index))))
        return h


class TriModalGAT(nn.Module):
    """三模态知识图谱融合提取器 + 逐节点分类头。

    mode: fused=三图共享注意力融合 | static/chain/network=单模态消融。
    """

    def __init__(self, vocab_sizes, dense_dim, emb_dim=16, hidden=128,
                 heads=4, layers=2, dropout=0.1, mode="fused"):
        super().__init__()
        self.mode = mode
        self.encoder = NodeEncoder(vocab_sizes, dense_dim, emb_dim, hidden, dropout)
        self.branches = nn.ModuleDict(
            {n: GATBranch(hidden, heads, layers, dropout) for n in ("static", "chain", "network")})
        self.fuse_score = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, dense, sparse, edges):
        h0 = self.encoder(dense, sparse)
        if self.mode == "fused":
            stack = torch.stack([self.branches[n](h0, edges[n])
                                 for n in ("static", "chain", "network")], dim=1)  # [N,3,H]
            alpha = torch.softmax(self.fuse_score(stack), dim=1)
            z = (alpha * stack).sum(dim=1)
        else:
            z = self.branches[self.mode](h0, edges[self.mode])
        return self.head(z).squeeze(-1), z
