"""链式 LSTM (官方口径) 配置中心。

严格对齐官方 Delay-data/Datasets/Flight_chain.py:
- 按 (OP_CARRIER, OP_CARRIER_FL_NUM, FL_DATE) 分组成链, 组内按计划起飞时间排序;
- 15 个静态特征 (dense 7 + sparse 8), 实际延误只作标签, 绝不进输入;
- max_len=6 截断/补零; 单向 RNN, 位置 t 只用 1..t (无未来泄露)。

数据复用外层 LLM/data 下已切分好的 train/val/test.csv (date-order 219/73/74),
链在各分片内构建 —— 同一 (carrier, flnum, date) 不跨分片, 无切分泄露。
"""
import os


# 本文件夹根 = 链式数据
LLM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 外层 LLM 根 = 模型搭建/LLM (复用其 data)
OUTER_LLM_ROOT = os.path.dirname(LLM_ROOT)

DATA_DIR = os.environ.get("LLM_DATA_DIR", os.path.join(OUTER_LLM_ROOT, "data"))
MODEL_OUTPUT_ROOT = os.environ.get("LLM_MODEL_OUTPUT_ROOT", os.path.join(LLM_ROOT, "模型"))
EXPERIMENT_NAME = os.environ.get("CHAIN_EXPERIMENT_NAME", "chain_lstm")
SAVE_DIR = os.path.join(MODEL_OUTPUT_ROOT, EXPERIMENT_NAME)
CACHE_DIR = os.path.join(SAVE_DIR, "cache")
MANIFEST_PATH = os.path.join(DATA_DIR, "dataset_manifest.csv")


# ---- 官方链式特征口径 (Flight_chain.py) ----
# 分组键: 一条链 = 同一航司+航班号+日期 的多段航段序列
GROUP_KEYS = ["OP_CARRIER", "OP_CARRIER_FL_NUM", "FL_DATE"]
SORT_KEY = "CRS_DEP_TIME_MIN"  # 组内按计划起飞时间升序

# dense 连续特征 (7): 天气 + 航班频次, 全静态
DENSE_COLS = ["O_TEMP", "D_TEMP", "O_PRCP", "D_PRCP", "O_WSPD", "D_WSPD", "FLIGHTS"]

# sparse 类别特征 (8): 月/星期/起降小时/起降机场/航司/航班号
# CRS_*_TIME_HOUR 由 CRS_*_TIME_MIN // 60 派生
SPARSE_COLS = [
    "FL_MONTH",
    "FL_WEEK",
    "CRS_ARR_TIME_HOUR",
    "CRS_DEP_TIME_HOUR",
    "ORIGIN_INDEX",
    "DEST_INDEX",
    "OP_CARRIER",
    "OP_CARRIER_FL_NUM",
]

# 目标: DEP_DELAY > 15min 为延误 (与全项目一致); 亦保留 ARR 便于对照
TARGET = os.environ.get("CHAIN_TARGET", "DEP")  # DEP | ARR
LABEL_THRESHOLD_MINUTES = 15.0
MAX_CHAIN_LEN = int(os.environ.get("CHAIN_MAX_LEN", 6))
CLASS_NAMES = {0: "normal", 1: "delayed"}


# ---- 采样规模 (可控, 便于在本机快速跑) ----
def _opt_int(name, default):
    value = os.environ.get(name, default)
    if value is None:
        return None
    if str(value).lower() in {"", "none", "null"}:
        return None
    return int(value)


MAX_TRAIN_CHAINS = _opt_int("CHAIN_MAX_TRAIN", "30000")
MAX_VAL_CHAINS = _opt_int("CHAIN_MAX_VAL", "15000")
MAX_TEST_CHAINS = _opt_int("CHAIN_MAX_TEST", "15000")
RANDOM_SEED = int(os.environ.get("CHAIN_SEED", 42))


# ---- RNN 结构 ----
RNN_TYPE = os.environ.get("CHAIN_RNN", "LSTM").upper()  # LSTM | GRU
# TCN 基线 (5_模型测试): 因果空洞卷积 kernel 大小 / 块数(空洞 1,2,4,...)
TCN_KERNEL = int(os.environ.get("CHAIN_TCN_KERNEL", 3))
TCN_LAYERS = int(os.environ.get("CHAIN_TCN_LAYERS", 3))
HIDDEN_SIZE = int(os.environ.get("CHAIN_HIDDEN", 128))
NUM_LAYERS = int(os.environ.get("CHAIN_LAYERS", 1))
DROPOUT = float(os.environ.get("CHAIN_DROPOUT", 0.1))
EMB_DIM = int(os.environ.get("CHAIN_EMB_DIM", 16))  # 每个 sparse 特征的 embedding 维度
EMB_DIM_MAX = int(os.environ.get("CHAIN_EMB_DIM_MAX", 32))  # 大基数列上限


# ---- 训练 ----
EPOCHS = int(os.environ.get("CHAIN_EPOCHS", 20))
BATCH_SIZE = int(os.environ.get("CHAIN_BATCH_SIZE", 256))
LEARNING_RATE = float(os.environ.get("CHAIN_LR", 1e-3))
WEIGHT_DECAY = float(os.environ.get("CHAIN_WEIGHT_DECAY", 1e-5))
# 类不平衡: BCE pos_weight (None=不加权, 贴合论文默认; 设数值可提升召回)
_pos_weight = os.environ.get("CHAIN_POS_WEIGHT", "none")
POS_WEIGHT = None if _pos_weight.lower() in {"", "none", "null"} else float(_pos_weight)
EARLY_STOP_PATIENCE = int(os.environ.get("CHAIN_PATIENCE", 5))


# ---- XGBoost 基线 (2_基准模型): 同 15 特征, 逐航班拍平, 无序列 ----
XGB_N_ESTIMATORS = int(os.environ.get("XGB_N_ESTIMATORS", 300))
XGB_MAX_DEPTH = int(os.environ.get("XGB_MAX_DEPTH", 6))
XGB_LEARNING_RATE = float(os.environ.get("XGB_LEARNING_RATE", 0.05))
XGB_SUBSAMPLE = float(os.environ.get("XGB_SUBSAMPLE", 0.9))
XGB_COLSAMPLE_BYTREE = float(os.environ.get("XGB_COLSAMPLE_BYTREE", 0.9))
XGB_N_JOBS = int(os.environ.get("XGB_N_JOBS", 8))


# ---- LLM (Gemma) 链上下文融合 (4_链式LLM融合) ----
# 复用外层同一个 Gemma-4-E4B 文本塔; 冻结, 嵌入一次性缓存。
GEMMA_MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH",
    r"C:\Users\16960\.cache\modelscope\hub\models\google\gemma-4-E4B",
)
LLM_PROMPT_VERSION = os.environ.get("CHAIN_LLM_PROMPT_VER", "chainctx_v1")
LLM_MAX_LEN = int(os.environ.get("CHAIN_LLM_MAX_LEN", 256))
LLM_EMBED_BATCH = int(os.environ.get("CHAIN_LLM_BATCH", 8))
LLM_MAX_PREV_LEGS = int(os.environ.get("CHAIN_LLM_MAX_PREV", 5))  # 提示里最多回溯几段前序航段

# 融合头
FUSION_MODE = os.environ.get("CHAIN_FUSION_MODE", "concat")  # concat | llm_only | lstm_only
FUSION_PROJ_DIM = int(os.environ.get("CHAIN_FUSION_PROJ", 128))  # LLM 嵌入投影维度
FUSION_HEAD_HIDDEN = int(os.environ.get("CHAIN_FUSION_HEAD", 128))
FUSION_DROPOUT = float(os.environ.get("CHAIN_FUSION_DROPOUT", 0.1))
FUSION_EPOCHS = int(os.environ.get("CHAIN_FUSION_EPOCHS", str(EPOCHS)))
FUSION_LR = float(os.environ.get("CHAIN_FUSION_LR", str(LEARNING_RATE)))

LLM_EMBED_CACHE_DIR = os.path.join(SAVE_DIR, "llm_embed_cache")


# ---- 深度融合 (5_深度融合): LSTM soft-token 注入 LLM + 输出门控 ----
# 复用 chain_lstm 缓存的链张量 + 文本; 结果写入同一 SAVE_DIR。
# 与浅融合不同: LLM 实时前向 (不缓存), 梯度回传到对齐层。
# 被注入的 LLM: 默认与 GEMMA_MODEL_PATH 同, 可指向更小模型 (如 gemma-4-E2B) 以可收敛。
DF_MODEL_PATH = os.environ.get("DF_MODEL_PATH", GEMMA_MODEL_PATH)
DF_MAX_TRAIN_CHAINS = _opt_int("DF_MAX_CHAINS", "2000")
DF_MAX_EVAL_CHAINS = _opt_int("DF_MAX_EVAL", "2000")
DF_EPOCHS = int(os.environ.get("DF_EPOCHS", 3))
DF_BATCH = int(os.environ.get("DF_BATCH", 8))
DF_LR = float(os.environ.get("DF_LR", 1e-3))
DF_GATE_DIM = int(os.environ.get("DF_GATE_DIM", 256))       # 门控融合公共维
DF_HEAD_HIDDEN = int(os.environ.get("DF_HEAD_HIDDEN", 128))
DF_PROMPT_MAX_LEN = int(os.environ.get("CHAIN_LLM_MAX_LEN", 160))  # 链级提示截断长度
DF_USE_LORA = os.environ.get("DF_USE_LORA", "0").lower() in {"1", "true", "yes"}
DF_LORA_R = int(os.environ.get("DF_LORA_R", 8))
DF_LORA_ALPHA = int(os.environ.get("DF_LORA_ALPHA", 16))
DF_EARLY_STOP_PATIENCE = int(os.environ.get("DF_PATIENCE", 4))
# 真注入专用: Time-LLM 式原型数 / 跨注意维 / 梯度检查点(省显存)
DF_NUM_PROTO = int(os.environ.get("DF_NUM_PROTO", 256))
DF_ATTN_DIM = int(os.environ.get("DF_ATTN_DIM", 128))
DF_GRAD_CHECKPOINT = os.environ.get("DF_GRAD_CHECKPOINT", "1").lower() in {"1", "true", "yes"}
# GPT4TS 配方: 冻结注意力+FFN, 只放开 LayerNorm 仿射参数 (配合可训的输入对齐层+输出头)
DF_TUNE_NORM = os.environ.get("DF_TUNE_NORM", "1").lower() in {"1", "true", "yes"}
# 消融开关: 真 -> 只用结构直连流(LLM流置零, 且跳过 LLM 前向), 作同口径纯结构基线
DF_ABLATE_LLM = os.environ.get("DF_ABLATE_LLM", "0").lower() in {"1", "true", "yes"}
# 深融合专用正则(抑制过拟合)
DF_DROPOUT = float(os.environ.get("DF_DROPOUT", 0.2))
DF_WEIGHT_DECAY = float(os.environ.get("DF_WEIGHT_DECAY", 0.01))


def data_path(split):
    return os.path.join(DATA_DIR, f"{split}.csv")


# ---- 真序列融合 (6_真序列融合): 机场近窗口计划航班序列 -> LLM ----
# 合规: 只取目标航班起飞前、同机场同日的计划航班, 仅静态字段, 无实际延误/实际时刻。
AIRSEQ_WINDOW_MIN = int(os.environ.get("AIRSEQ_WINDOW_MIN", 120))     # 回看窗口(分钟)
AIRSEQ_MAX_NEIGHBORS = int(os.environ.get("AIRSEQ_MAX_NB", 16))       # 最多邻居航班数
AIRSEQ_PROMPT_VERSION = os.environ.get("AIRSEQ_PROMPT_VER", "airseq_v1")
AIRSEQ_MAX_LEN = int(os.environ.get("AIRSEQ_MAX_LEN", 384))           # 序列提示较长, 截断长度大一些
AIRSEQ_EPOCHS = int(os.environ.get("AIRSEQ_EPOCHS", str(EPOCHS)))
# 跨注意力读出 (readout=crossattn): 按邻居切片池化, h_t 跨注意力邻居序列
AIRSEQ_SEQ_PROMPT_VERSION = os.environ.get("AIRSEQ_SEQ_PROMPT_VER", "airseq_seq_v1")
AIRSEQ_CA_DIM = int(os.environ.get("AIRSEQ_CA_DIM", 128))
AIRSEQ_CA_HEADS = int(os.environ.get("AIRSEQ_CA_HEADS", 4))


def airseq_texts_path(split):
    return os.path.join(CACHE_DIR, f"airseq_{split}.pkl")


def cache_path(split):
    return os.path.join(CACHE_DIR, f"chain_{split}.pt")


def texts_path(split):
    return os.path.join(CACHE_DIR, f"texts_{split}.pkl")


def encoders_path():
    return os.path.join(CACHE_DIR, "encoders.pkl")
