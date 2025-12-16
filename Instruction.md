既然你是在**没有图形界面的 Linux 服务器**（Headless Server）上操作，我们就不能依赖 LLaMA-Factory 的 WebUI 了。

我们需要使用 **命令行 (CLI)** 模式。别担心，CLI 其实更稳定，而且显得你更专业（Linux 老手都是敲命令的）。

你的显卡是 **24G (RTX 3090/4090)**，非常适合跑 **Qwen2.5-7B** 的 **LoRA 微调**。

以下是完整的一步步操作指南：

-----

### 第一步：环境准备 (LLaMA-Factory)

如果你还没安装 LLaMA-Factory，请先执行这些命令：

```bash
# 1. 克隆代码库
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 2. 创建虚拟环境 (强烈建议，防止污染环境)
conda create -n llama_factory python=3.10 -y
conda activate llama_factory

# 3. 安装依赖 (这一步可能需要几分钟)
pip install -e .[metrics]
pip install bitsandbytes # 量化微调必须组件
```

-----

### 第二步：注册你的数据集 (关键！)

LLaMA-Factory 不会自己扫描文件夹，你需要告诉它你的 JSON 文件叫什么。

1.  **上传数据**：
    把你的 `steam_sft_train.json` 上传到 `LLaMA-Factory/data/` 目录下。

2.  **编辑注册表**：
    使用 `vim` 或 `nano` 编辑 `data/dataset_info.json` 文件：

    ```bash
    vim data/dataset_info.json
    ```

3.  **添加配置**：
    在文件的 JSON 对象中，添加一段你的数据配置（注意不要破坏 JSON 格式，上一项后面要加逗号）：

    ```json
    "steam_reviews": {
      "file_name": "steam_sft_train.json",
      "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output"
      }
    }
    ```

    *解释：这里告诉框架，我的数据集代号叫 `steam_reviews`，对应的文件是 `steam_sft_train.json`，JSON 里的 instruction 对应 prompt，input 对应 query，output 对应 response。*

-----

### 第三步：构造训练命令 (The Magic Command)

这是最核心的一步。针对 **24G 显存**，我们需要开启 **4-bit 量化 (QLoRA)** 来节省显存，同时把 Batch Size 设大一点以保证速度。

请直接在 `LLaMA-Factory` 目录下运行以下命令（建议复制保存为一个 `run_train.sh` 脚本运行）：

```bash
# 确保你在 LLaMA-Factory 目录下
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset steam_reviews \
    --dataset_dir data \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --output_dir saves/qwen2.5-7b-steam-lora \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 3.0 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_ratio 0.1 \
    --fp16 True \
    --quantization_bit 4
```

#### 🛠️ 参数详解（答辩时可以用）：

  * `--model_name_or_path`: 使用 Qwen2.5-7B-Instruct 作为基座（如果你还没下载，它会自动从 HuggingFace/ModelScope 下载）。
  * `--dataset steam_reviews`: 刚才注册的数据集代号。
  * `--lora_target all`: 对模型的所有层进行微调，效果比默认好，虽然慢一点点。
  * `--quantization_bit 4`: **核心！** 4bit 量化加载。这样模型只占 6GB 显存，剩下的 18GB 都可以用来放训练数据，防止 OOM (Out of Memory)。
  * `--per_device_train_batch_size 4` \* `--gradient_accumulation_steps 4`: 实际 Batch Size = 16。这是一个比较稳健的数值。
  * `--learning_rate 2e-4`: LoRA 微调的标准学习率。

-----

### 第四步：监控训练过程

当你运行上述命令后，终端会开始疯狂跳动 Loss 值。

1.  **看 Loss**：
    你应该看到 `loss` 从一开始的 `2.x` 或 `1.x` 慢慢下降。

      * 如果降到 `0.8` 以下，说明它学会说话了。
      * 如果降到 `0.1` 以下，可能过拟合了（不过对于风格迁移，过拟合一点没关系）。

2.  **防止断连**：
    如果你是用 SSH 连的服务器，建议用 `tmux` 或者 `nohup` 运行，否则网一断训练就停了。

    ```bash
    # 后台运行示例
    nohup bash run_train.sh > train.log 2>&1 &
    # 查看日志
    tail -f train.log
    ```

-----

### 第五步：在终端里测试模型 (Inference)

训练大概需要 1-3 小时（取决于你的数据量）。训练完后，不需要导出模型，可以直接加载 LoRA 权重在终端里测试！

使用 `llamafactory-cli chat` 命令：

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path saves/qwen2.5-7b-steam-lora \
    --template qwen \
    --quantization_bit 4
```

**运行后，你会进入一个像 Python input 的界面：**

  * **User**: 请以资深玩家的身份，点评一下《黑神话：悟空》这款游戏。
  * **Assistant**: (这里模型应该会输出你训练的那些骚话，比如“打击感爆棚”、“空气墙恶心”之类的内容)

-----

### 💡 给你的 Dify 部署建议 (API 服务)

等你在终端里测试满意了，你需要把它变成一个 API 给 Dify 调用。
LLaMA-Factory 自带了 OpenAI 格式的 API 服务！

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli api \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path saves/qwen2.5-7b-steam-lora \
    --template qwen \
    --quantization_bit 4 \
    --api_port 8000
```

运行这个命令后，你的服务器 `8000` 端口就开启了一个类似 OpenAI 的接口。
在 Dify 里配置：

  * **模型提供商**: OpenAI-API-Compatible
  * **API URL**: `http://你的服务器IP:8000/v1`
  * **API Key**: (随便填，例如 `sk-123`)

这样，你的 Dify 智能体就能用上这个\*\*“满嘴骚话”的老玩家模型\*\*了！🎮



export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download Qwen/Qwen2.5-7B-Instruct --local-dir ./hf_cache/Qwen2.5-7B-Instruct




on 3080ti 10G
新增 --gradient_checkpointing True (最重要)：

作用：大幅降低显存占用（约节省 50%-60%）。它不保存所有中间激活层，而是在反向传播时重算。

效果：没有这个参数，10G 显存跑 4B 模型即使是 4-bit 量化也非常危险。加上后，显存占用会变得很平稳。

XXX 新增 --cutoff_len 1024 (解决"后来OOM"的关键)：

作用：强制将超过 1024 个 token 的评论截断。

原因：你之前的报错是因为遇到了某条特别长的 Steam 评论。如果不限制长度，显存需求会随长度呈平方级或线性增长，瞬间击穿 10G 显存墙。对于评论分析，1024 长度通常完全够用了。

调整 Batch Size 策略：

原配置：Batch=4, Accumulation=4 (总Batch=16)

新配置：--per_device_train_batch_size 2, --gradient_accumulation_steps 8 (总Batch=16)

原因：将单次喂给显卡的样本数从 4 降到 2，可以显著降低显存峰值。通过增加“梯度累积步数”到 8，保持了原本的训练效果（总 Batch 依然是 16），模型收敛速度和效果不会变，但显存更安全。




LLaMA-Factory 支持从上一次保存的检查点（Checkpoint）断点续训。

方法：添加 --resume_from_checkpoint 参数
你只需要在运行脚本的命令中加入一行参数即可。

方式：自动恢复
添加 --resume_from_checkpoint True。 程序会自动去你的 --output_dir（即 saves/qwen3-4b-steam-lora）目录下寻找最新的 checkpoint-xxx 文件夹，并加载模型权重、优化器状态和学习率进度，从那一刻继续训练。

Bash

# 修改后的 run_train_Qwen3_4B.sh
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    ... （其他参数保持不变） ...
    --output_dir saves/qwen3-4b-steam-lora \
    --resume_from_checkpoint True \
    ...