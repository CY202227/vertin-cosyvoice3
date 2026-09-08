# Vertin CosyVoice3

Vertin英语声音：CosyVoice3 LLM 微调 + 本地推理。仓库只分三个目录。

```text
finetune/   数据准备、SFT 训练脚本、checkpoint
infer/      输入文本出语音
runtime/    CosyVoice 源码 + Fun-CosyVoice3 底模
```

`.venv/` 是本机环境，不要上传。

## GitHub 传什么

GitHub 单文件上限 100MB，不要传权重和长音频。`.gitignore` 已排除：

- `runtime/pretrained_models/`（约 9GB）
- `finetune/ckpts/*.pt`（每个约 1GB）
- `finetune/data/` 里的 wav / 源视频
- `infer/outputs/`
- `.venv/`

会上传：训练/推理脚本、`conf`、`infer/assets/prompt.wav`（很短的参考音）、`infer/examples/story.txt`。

别人克隆后需要自己准备：

1. 克隆 CosyVoice 到 `runtime/CosyVoice`（若尚未带上）
2. 下载底模到 `runtime/pretrained_models/Fun-CosyVoice3-0.5B`
3. 放入 `finetune/ckpts/epoch_8_whole.pt`

```powershell
python runtime/download_cosyvoice3.py
```

## 推理

需要 `transformers==4.51.3`。

```powershell
cd H:\dev\voice_maker
$env:PYTHONPATH = "H:\dev\voice_maker\runtime\CosyVoice;H:\dev\voice_maker\runtime\CosyVoice\third_party\Matcha-TTS"
.\.venv\Scripts\python.exe infer\infer.py "Hi, Sonetto. How do you feel?"
.\.venv\Scripts\python.exe infer\infer.py -f infer\examples\story.txt -o infer\outputs\story.wav
```

默认输出 `infer/outputs/tts.wav`。语速跟模型原速；若要微调用 `--speed`（大于 1 更快）。

## 微调

在 DGX Spark 上：

```bash
cd finetune/spark
source env.sh
bash setup_cosyvoice3.sh
bash prepare_parquet.sh
bash train_llm.sh
```

Windows 上只做数据准备：`finetune/scripts/`。
