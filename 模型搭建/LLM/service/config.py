"""LoRA 微调配置"""
import os

# 路径
MODEL_PATH = r"C:\Users\16960\.cache\modelscope\hub\models\google\gemma-4-E4B"
DATA_DIR = r"c:\Users\16960\Desktop\期末论文\模型搭建\LLM\data"
SAVE_DIR = r"c:\Users\16960\Desktop\期末论文\模型搭建\LLM\模型\lora_adapter"

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
# 梯度累积: 等效 batch_size = 1 * 32 = 32，保持原 smoke test 梯度规模
GRAD_ACCUM = 32
# 学习率: 第一轮先用更稳的 1e-4
LEARNING_RATE = 1e-4
# 训练轮数: 只跑 1 轮做 smoke test
EPOCHS = 1
# 第一轮 smoke test 只读少量样本, 确认链路和 loss 趋势后再改成 None 跑全量
TRAIN_MAX_SAMPLES = 2000
VAL_MAX_SAMPLES = 2000
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
SMOKE_NUM_SAMPLES = 2000
BALANCED_POS_RATIO = 0.20
# full: 保留原始字段文本; compact: 重新整理为更短、更稳定的结构化文本
PROMPT_STYLE = "compact"
# 评估时每次读取多少条样本; None 表示全量
EVAL_MAX_SAMPLES = 2000

# 数据接口:
# 旧格式: {"text": "...", "label": 0/1}
# 新格式: {"instruction": "...", "input": "...", "output": "...", "label": 0/1}
PROMPT_TEMPLATE_FULL = """任务：根据以下航班链信息，判断当前航班是否会出发延误超过15分钟。

要求：
1. 只能输出一个标签。
2. 不要输出解释、原因或多余文字。
3. 合法标签只有：正常、延误。

航班链信息：
{text}

答案："""

PROMPT_TEMPLATE_COMPACT = """任务：判断当前航班是否会出发延误超过15分钟。
只输出一个标签：正常 或 延误。

{text}

答案："""

PROMPT_TEMPLATES = {
    "full": PROMPT_TEMPLATE_FULL,
    "compact": PROMPT_TEMPLATE_COMPACT,
}
PROMPT_TEMPLATE = PROMPT_TEMPLATES[PROMPT_STYLE]
LABEL_MAP = {0: "正常", 1: "延误"}
