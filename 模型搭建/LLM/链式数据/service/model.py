"""链式序列模型: 单向 LSTM/GRU 的逐位置延误分类器 (官方口径)。

- 每个 sparse 列过 Embedding(padding_idx=0), 与标准化 dense 拼接成每步输入;
- 单向 RNN: 位置 t 只用 1..t 的静态特征 (无未来泄露);
- pack_padded_sequence 按 valid_len 屏蔽补零位置;
- 逐位置输出 logit, 预测该航班 DEP_DELAY>15。
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import SPARSE_COLS


class _Chomp1d(nn.Module):
    """裁掉因果卷积右侧多余 padding, 保证输出位置 t 只依赖 <=t 的输入 (无未来泄露)。"""

    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp] if self.chomp > 0 else x


class _CausalConvBlock(nn.Module):
    """TCN 残差块: 两层空洞因果卷积 (左侧 padding + chomp), 不做跨时间归一化 (防归一化泄露)。"""

    def __init__(self, ch, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.chomp1 = _Chomp1d(pad)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.chomp2 = _Chomp1d(pad)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # [B,C,T]
        y = self.drop(self.act(self.chomp1(self.conv1(x))))
        y = self.drop(self.act(self.chomp2(self.conv2(y))))
        return self.act(y + x)


class ChainTCN(nn.Module):
    """链式 TCN 基线: 因果空洞卷积, 与 ChainRNN 同口径 (逐位置、无未来泄露、masked BCE)。

    只左侧 padding + chomp => 位置 t 只看 1..t; padding 在序列末尾, 有效位置不读 padding。
    """

    def __init__(self, vocab_sizes, dense_dim, emb_dim, hidden, num_layers, dropout, kernel=3):
        super().__init__()
        self.embeds = nn.ModuleList(
            [nn.Embedding(int(vocab_sizes[col]), emb_dim, padding_idx=0) for col in SPARSE_COLS]
        )
        input_dim = emb_dim * len(SPARSE_COLS) + dense_dim
        self.inp = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(
            [_CausalConvBlock(hidden, kernel, 2 ** i, dropout) for i in range(num_layers)]
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, dense, sparse, valid_len):
        embs = [emb(sparse[:, :, i]) for i, emb in enumerate(self.embeds)]
        x = torch.cat(embs + [dense], dim=-1)       # [B,L,input_dim]
        x = self.inp(x).transpose(1, 2)              # [B,hidden,L]
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2)                        # [B,L,hidden]
        return self.head(x).squeeze(-1)              # [B,L]


class ChainRNN(nn.Module):
    def __init__(self, vocab_sizes, dense_dim, emb_dim, hidden, num_layers, dropout, rnn_type="LSTM"):
        super().__init__()
        self.embeds = nn.ModuleList(
            [nn.Embedding(int(vocab_sizes[col]), emb_dim, padding_idx=0) for col in SPARSE_COLS]
        )
        input_dim = emb_dim * len(SPARSE_COLS) + dense_dim
        rnn_cls = nn.GRU if rnn_type.upper() == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, dense, sparse, valid_len):
        # dense [B,L,Dd], sparse [B,L,Ds] (int64), valid_len [B]
        embs = [emb(sparse[:, :, i]) for i, emb in enumerate(self.embeds)]
        x = torch.cat(embs + [dense], dim=-1)  # [B,L,input_dim]

        lengths = valid_len.clamp(min=1).to("cpu")
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        logits = self.head(out).squeeze(-1)  # [B,L]
        return logits


class ChainFusionRNN(nn.Module):
    """真融合: 链嵌入 h_t (LSTM) 与 LLM 链上下文嵌入投影 e_t 拼接, 逐位置分类。

    对齐论文框架 (framework.png): Features = concat(链嵌入, [图嵌入], ...) -> 分类头。
    这里把 LLM 语义表征作为又一路 concat 分支; 端到端联合训练 (LSTM/投影/头可训, LLM 冻结)。
    mode: concat=链+LLM | lstm_only=仅链 | llm_only=仅LLM (消融)。
    """

    def __init__(self, vocab_sizes, dense_dim, emb_dim, hidden, num_layers, dropout, rnn_type,
                 llm_dim, proj_dim, head_hidden, mode="concat"):
        super().__init__()
        self.mode = mode
        self.embeds = nn.ModuleList(
            [nn.Embedding(int(vocab_sizes[col]), emb_dim, padding_idx=0) for col in SPARSE_COLS]
        )
        input_dim = emb_dim * len(SPARSE_COLS) + dense_dim
        rnn_cls = nn.GRU if rnn_type.upper() == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=False,
        )
        self.llm_norm = nn.LayerNorm(llm_dim)
        self.llm_proj = nn.Sequential(
            nn.Linear(llm_dim, proj_dim), nn.GELU(), nn.Dropout(dropout)
        )
        if mode == "concat":
            head_in = hidden + proj_dim
        elif mode == "lstm_only":
            head_in = hidden
        elif mode == "llm_only":
            head_in = proj_dim
        else:
            raise ValueError(f"unknown fusion mode: {mode}")
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(dropout), nn.Linear(head_in, head_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def _encode_seq(self, dense, sparse, valid_len):
        embs = [emb(sparse[:, :, i]) for i, emb in enumerate(self.embeds)]
        x = torch.cat(embs + [dense], dim=-1)
        lengths = valid_len.clamp(min=1).to("cpu")
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return out  # [B,L,hidden]

    def forward(self, dense, sparse, valid_len, llm_emb):
        parts = []
        if self.mode in ("concat", "lstm_only"):
            parts.append(self._encode_seq(dense, sparse, valid_len))
        if self.mode in ("concat", "llm_only"):
            parts.append(self.llm_proj(self.llm_norm(llm_emb.float())))
        feat = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        logits = self.head(feat).squeeze(-1)  # [B,L]
        return logits


class AirseqCrossAttn(nn.Module):
    """真序列跨注意力融合: LSTM 的 h_t 作 query 对邻居序列(key/value)做多头跨注意力。

    不再把序列压成单一向量: h_t 自己去“挑”该看哪几条邻居航班; summary 作残差。
    mode: crossattn=结构+跨注意力邻居+summary | lstm_only=仅结构 (同口径消融)。
    """

    def __init__(self, vocab_sizes, dense_dim, emb_dim, hidden, num_layers, dropout, rnn_type,
                 llm_dim, ca_dim, ca_heads, head_hidden, mode="crossattn"):
        super().__init__()
        self.mode = mode
        self.ca_dim = ca_dim
        self.ca_heads = ca_heads
        assert ca_dim % ca_heads == 0, "ca_dim 必须能被 ca_heads 整除"
        self.embeds = nn.ModuleList(
            [nn.Embedding(int(vocab_sizes[col]), emb_dim, padding_idx=0) for col in SPARSE_COLS]
        )
        input_dim = emb_dim * len(SPARSE_COLS) + dense_dim
        rnn_cls = nn.GRU if rnn_type.upper() == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=False,
        )
        self.llm_norm = nn.LayerNorm(llm_dim)
        self.q_proj = nn.Linear(hidden, ca_dim)
        self.k_proj = nn.Linear(llm_dim, ca_dim)
        self.v_proj = nn.Linear(llm_dim, ca_dim)
        self.s_proj = nn.Sequential(nn.Linear(llm_dim, ca_dim), nn.GELU(), nn.Dropout(dropout))
        if mode == "crossattn":
            head_in = hidden + ca_dim + ca_dim   # 结构 + 邻居跨注意力上下文 + summary 残差
        elif mode == "catgate":
            # 跨注意力+concat+summary 后, 再加一层特征级门控 (crossattn 与 gate 的叠合版)
            head_in = hidden + ca_dim + ca_dim
            self.cat_gate = nn.Linear(head_in, head_in)
        elif mode == "gate":
            # 残差门控: ctx 升维后与 h 相加, 门控 g 控制 LLM 贡献 (MSGCA/T3Time 式)
            self.ctx_up = nn.Linear(ca_dim, hidden)
            self.gate = nn.Linear(hidden + ca_dim, hidden)
            head_in = hidden
        elif mode == "lstm_only":
            head_in = hidden
        else:
            raise ValueError(f"unknown airseq mode: {mode}")
        self.head = nn.Sequential(
            nn.LayerNorm(head_in), nn.Dropout(dropout),
            nn.Linear(head_in, head_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def _encode_seq(self, dense, sparse, valid_len):
        embs = [emb(sparse[:, :, i]) for i, emb in enumerate(self.embeds)]
        x = torch.cat(embs + [dense], dim=-1)
        lengths = valid_len.clamp(min=1).to("cpu")
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return out  # [B,L,hidden]

    def _cross_attend(self, h, neighbors, nb_mask):
        # h [B,L,hidden]; neighbors [B,L,K,llm]; nb_mask [B,L,K]
        B, L, K, _ = neighbors.shape
        nh, dh = self.ca_heads, self.ca_dim // self.ca_heads
        nb = self.llm_norm(neighbors.float())
        q = self.q_proj(h).view(B, L, nh, dh)
        k = self.k_proj(nb).view(B, L, K, nh, dh).permute(0, 1, 3, 2, 4)   # [B,L,nh,K,dh]
        v = self.v_proj(nb).view(B, L, K, nh, dh).permute(0, 1, 3, 2, 4)
        scores = torch.einsum("blhd,blhkd->blhk", q, k) / (dh ** 0.5)       # [B,L,nh,K]
        mask = nb_mask[:, :, None, :]                                      # [B,L,1,K]
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("blhk,blhkd->blhd", attn, v).reshape(B, L, self.ca_dim)
        valid = (nb_mask.sum(-1, keepdim=True) > 0).float()                # 无邻居位置上下文置零
        return ctx * valid

    def forward(self, dense, sparse, valid_len, summary, neighbors, nb_mask):
        h = self._encode_seq(dense, sparse, valid_len)
        if self.mode == "lstm_only":
            return self.head(h).squeeze(-1)
        ctx = self._cross_attend(h, neighbors, nb_mask)
        if self.mode == "gate":
            up = self.ctx_up(ctx)                                        # [B,L,hidden]
            g = torch.sigmoid(self.gate(torch.cat([h, ctx], dim=-1)))    # 逐维门
            return self.head(h + g * up).squeeze(-1)                     # 残差: 结构主导 + 门控LLM增量
        s = self.s_proj(self.llm_norm(summary.float()))
        feat = torch.cat([h, ctx, s], dim=-1)
        if self.mode == "catgate":
            g = torch.sigmoid(self.cat_gate(feat))                       # 特征级门控: 逐维控信多少
            feat = g * feat
        return self.head(feat).squeeze(-1)
