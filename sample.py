from unsloth import FastLanguageModel
import torch

# 设置最大序列长度
max_seq_length = 2048  # 可以选择任何长度，unsloth 內部支持 RoPE 伸缩

dtype = None  # 数据类型，None 代表自动检测。Tesla T4、V100 适合 float16，Ampere+ 适合 bfloat16

# 是否使用 4bit 量化以减少显存占用
load_in_4bit = True  # 设为 False 可以关闭 4bit 量化

# 预先量化的 4bit 模型列表，这些模型下载速度更快且减少 OOM（内存溢出）
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # Mistral v3，速度提升 2 倍
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3，15 万亿训练数据，速度提升 2 倍
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3，速度提升 2 倍
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma，速度提升 2.2 倍
] # 更多模型请访问 https://huggingface.co/unsloth

# 加载 LLaMA-3 8B 4bit 量化模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",  # 选择要加载的模型
    max_seq_length = max_seq_length,  # 设置最大序列长度
    dtype = dtype,  # 设置数据类型
    load_in_4bit = load_in_4bit,  # 是否使用 4bit 量化
)

# 配置 LoRA（低秩适配）参数进行微调
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA 低秩矩阵的秩，建议使用 8、16、32、64、128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,  # LoRA alpha 参数
    lora_dropout = 0,  # LoRA dropout 设置，0 表示优化的设置
    bias = "none",  # 是否使用偏置参数，"none" 代表优化设置
    use_gradient_checkpointing = "unsloth",  # 使用梯度检查点节省显存，适用于长上下文
    random_state = 3407,  # 随机种子，保证实验可复现
    use_rslora = False,  # 是否使用 Rank Stabilized LoRA
    loftq_config = None,  # 是否使用 LoftQ 量化
)

from datasets import load_dataset

# 加载 Alpaca-GPT4 数据集（用于指令微调）
dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
print(dataset.column_names)  # 输出数据集的列名

from unsloth import to_sharegpt

# 转换数据集格式为 ShareGPT 格式
dataset = to_sharegpt(
    dataset,
    merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
    output_column_name="output",
    conversation_extension=3,  # 处理较长的对话
)

from unsloth import standardize_sharegpt

# 标准化 ShareGPT 数据集格式
dataset = standardize_sharegpt(dataset)

# 定义对话模板
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

from unsloth import apply_chat_template

# 应用聊天模板到数据集
dataset = apply_chat_template(
    dataset,
    tokenizer=tokenizer,
    chat_template=chat_template,
)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# 配置训练参数
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,  # 设置为 True 可加速短序列训练
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

# 显示 GPU 资源占用情况
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# 开始训练
trainer_stats = trainer.train()

# 显示训练完成后的 GPU 资源占用情况
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
print(f"Peak reserved memory = {used_memory} GB.")

# 启用推理优化
FastLanguageModel.for_inference(model)

# 保存模型到本地
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# 重新加载训练好的 LoRA 模型
if False:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
pass  # 占位符，当前代码块无实际操作

messages = [  # 定义对话消息，用户输入一个斐波那契数列前几项，期待模型描述其特殊性
    {"role": "user", "content": "Describe anything special about a sequence. Your input is 1, 1, 2, 3, 5, 8,"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,  # 添加生成提示
    return_tensors="pt",  # 返回 PyTorch 张量
).to("cuda")  # 将张量移动到 GPU

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)  # 初始化文本流式输出
_ = model.generate(input_ids, streamer=text_streamer, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)

if False:
    # 强烈不建议这样做，建议使用 Unsloth 进行优化
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        "lora_model",  # 训练使用的 Lora 微调模型
        load_in_4bit=load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("lora_model")

# 保存模型到 8bit Q8_0 量化格式
if True:
    model.save_pretrained_gguf("model", tokenizer,)

# 提示用户获取 Hugging Face 令牌，并修改 "hf" 为自己的用户名
# 获取令牌地址：https://huggingface.co/settings/tokens
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, token="")

# 保存模型到 16bit GGUF 量化格式
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="f16", token="")

# 保存模型到 q4_k_m GGUF 量化格式
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="q4_k_m", token="")

# 以多种 GGUF 量化格式保存模型（适用于需要多种量化格式的情况）
if False:
    model.push_to_hub_gguf(
        "hf/model",  # 请将 "hf" 替换为你的 Hugging Face 用户名
        tokenizer,
        quantization_method=["q4_k_m", "q8_0", "q5_k_m"],  # 多种量化方法
        token="",  # 访问 https://huggingface.co/settings/tokens 获取令牌
    )

import subprocess

# 启动 Ollama 服务器
subprocess.Popen(["ollama", "serve"])
import time

time.sleep(3)  # 等待几秒钟以确保 Ollama 正常加载

print(tokenizer._ollama_modelfile)  # 输出 Ollama 模型文件路径


########### Begin of Modelfile #################
# 创建 Ollama 模型
# FROM {__FILE_LOCATION__}
# TEMPLATE """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.{{ if .Prompt }}

# ### Instruction:
# {{ .Prompt }}{{ end }}

# ### Response:
# {{ .Response }}<|end_of_text|>"""

# PARAMETER stop "<|eot_id|>"
# PARAMETER stop "<|start_header_id|>"
# PARAMETER stop "<|end_header_id|>"
# PARAMETER stop "<|end_of_text|>"
# PARAMETER stop "<|reserved_special_token_"
# PARAMETER temperature 1.5
# PARAMETER min_p 0.1
###########  END Of Modelfile  ############

# create an Ollama model called unsloth_model using the Modelfile which we auto generated!
# ollama create unsloth_model -f ./model/Modelfile
# ollama run unsloth_model  # 运行 Ollama 模型
