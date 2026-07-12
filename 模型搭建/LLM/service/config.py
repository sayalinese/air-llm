"""LoRA 微调配置"""
import os

# 路径
MODEL_PATH = r"C:\Users\16960\.cache\modelscope\hub\models\google\gemma-4-E4B"
DATA_DIR = r"c:\Users\16960\Desktop\期末论文\模型搭建\LLM\data_t60"
MODEL_OUTPUT_ROOT = r"c:\Users\16960\Desktop\期末论文\模型搭建\LLM\模型"
# 消融阶段可由 run_ablation.py 通过环境变量切换。
ABLATION_STAGES = ("current", "chain", "context", "nearby", "risk")
ABLATION_STAGE = os.environ.get("LLM_ABLATION_STAGE", "nearby")
if ABLATION_STAGE not in ABLATION_STAGES:
    raise ValueError(f"Unknown LLM_ABLATION_STAGE={ABLATION_STAGE!r}; expected one of {ABLATION_STAGES}")
EXPERIMENT_NAME = os.environ.get("LLM_EXPERIMENT_NAME", f"t60_probe_ablation_{ABLATION_STAGE}")
SAVE_DIR = os.path.join(MODEL_OUTPUT_ROOT, EXPERIMENT_NAME)

EXPECTED_SCHEMA_VERSION = "chain_llm_t60"
EXPECTED_PROMPT_VERSION = "propagation_capsule_t60_operational"
EXPECTED_OBSERVATION_POLICY = "utc_actual_event_at_or_before_t_minus_60"
EXPECTED_PREDICTION_HORIZON_MINUTES = 60

# 当前 modelscope 缓存是多模态权重, 但训练只需要文本塔。
# text_from_multimodal: 只抽取 model.language_model.* 加载到 Gemma4ForCausalLM。
MODEL_ARCH = "text_from_multimodal"

# LoRA 秩: 低显存 smoke test 先用 4
LORA_R = 4
# LoRA 缩放系数: 通常设为 R 的 2 倍, 控制适配器对原始权重的影响幅度
LORA_ALPHA = 8
# LoRA dropout: 防止适配器过拟合, 0.05 足够
LORA_DROPOUT = 0.05
# 低显存第一轮只注入注意力 q/v; 稳定后再扩到 k/o 或 MLP
LORA_TARGET = ["q_proj", "v_proj"]
# 多模态 Gemma4 权重下只给文本 language_model 注入 LoRA, 避免视觉/音频塔被误训练
LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

# 训练（第一轮极短测试版）
# 目标：先快速验证 Gemma base + LoRA 是否能学会稳定输出“正常/延误”
# 批次大小: 最低显存 smoke test
BATCH_SIZE = 1
# 梯度累积: 等效 batch_size = 16
GRAD_ACCUM = 16
# 学习率: 第一轮先用更稳的 1e-4
LEARNING_RATE = 1e-4
# probe 消融跑两轮；10k 候选跑一轮。
DATASET_VARIANT = os.environ.get("LLM_DATASET_VARIANT", "probe")
EPOCHS = int(os.environ.get("LLM_EPOCHS", 1 if DATASET_VARIANT == "medium" else 2))
DATA_FILES = {
    "full": {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"},
    "probe": {"train": "train_probe.jsonl", "val": "val_probe.jsonl", "test": "test_probe.jsonl"},
    "medium": {"train": "train_10k.jsonl", "val": "val_2k.jsonl", "test": "test_final.jsonl"},
}
if DATASET_VARIANT not in DATA_FILES:
    raise ValueError(f"Unknown LLM_DATASET_VARIANT={DATASET_VARIANT!r}; expected one of {tuple(DATA_FILES)}")
TRAIN_MAX_SAMPLES = None
VAL_MAX_SAMPLES = None
# 预热比例: 极短测试版缩短 warmup
WARMUP_RATIO = 0.03
# 最大序列长度: 低显存 smoke test
MAX_LEN = 192
# 权重衰减: 保持轻度正则化
WEIGHT_DECAY = 0.01

# 实验设置
# full: 使用完整训练集; balanced_smoke: 使用带权采样做类别均衡 smoke test
TRAIN_SAMPLING_MODE = "balanced_smoke"
# balanced_smoke 模式下每个 epoch 最多抽多少条样本, 设为 None 则等同完整训练集长度
SMOKE_NUM_SAMPLES = int(os.environ.get("LLM_SMOKE_NUM_SAMPLES", 10000 if DATASET_VARIANT == "medium" else 2048))
BALANCED_POS_RATIO = 0.20
# full: 保留原始字段文本; compact: 重新整理为更短、更稳定的结构化文本
PROMPT_STYLE = "compact"
# 评估时每次读取多少条样本; None 表示全量
EVAL_MAX_SAMPLES = None
EVAL_BATCH_SIZE = 8
RANDOM_SEED = 42
FAIL_ON_PROMPT_TRUNCATION = True

# 数据接口:
# 旧格式: {"text": "...", "label": 0/1}
# 新格式: {"instruction": "...", "input": "...", "output": "...", "label": 0/1}
PROMPT_TEMPLATE_FULL = """出延>15? A正常 B延误。答A/B。
{text}
答:"""

PROMPT_TEMPLATE_COMPACT = """出延>15? A正常 B延误。答A/B。
{text}
答:"""

PROMPT_TEMPLATES = {
    "full": PROMPT_TEMPLATE_FULL,
    "compact": PROMPT_TEMPLATE_COMPACT,
}
PROMPT_TEMPLATE = PROMPT_TEMPLATES[PROMPT_STYLE]
LABEL_MAP = {0: "A", 1: "B"}
CLASS_NAMES = {0: "正常", 1: "延误"}


def data_path(split):
    return os.path.join(DATA_DIR, DATA_FILES[DATASET_VARIANT][split])
