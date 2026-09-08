# 明天训练备忘 · 2026-09-08

在 **DGX Spark** 上做，不要在笔记本 3060 上训。本机只负责听评和改文本。

## 现状（别重复做）

- CosyVoice3 **只训过 LLM**。`finetune/ckpts/epoch_8_whole.pt` 最好（CV loss 4.809，step 492）。epoch 9 起过拟合。
- **flow / HiFT 没训**，继续用官方底模。官方脚本也写了：先不要训 flow。
- 语速：LLM 已经能学 token 时长。上次故事又慢又卡是推理里 librosa 拉伸，不是没学到语速。`infer/infer.py` 已关掉拉伸。

## 明天真正要训的

**第二轮 LLM SFT**，但必须先洗数据。现在 jsonl 是 FunASR 切的，单词从中间断开（`Reg ulus`、`Son ato`、`Ar can um`、`F ifty`），模型和语速都会被带偏。

### 1. 洗文本 / 重切（先做完再训）

- 打开 `finetune/data/processed/manifests/all.jsonl`
- 修专有名词：Regulus、Sonetto、Arcanum、Manus、Pavlov
- 丢掉从单词中间切开、或明显不是完整句的 clip
- 尽量保持 5–15 秒、文本和 wav 对得上
- 然后重新导出 Kaldi + parquet：

```bash
cd finetune
python scripts/04_export_kaldi.py
bash spark/prepare_parquet.sh
```

### 2. 再训 LLM（从官方 `llm.pt` 重新 SFT，不要从 epoch_8 接着训）

```bash
cd finetune/spark
source env.sh
bash train_llm.sh
```

- 配置已是 `lr 1e-5`、`constantlr`、`max_epoch 12`、`use_spk_embedding True`
- 看 TensorBoard / `cv_metrics`，**在 CV loss 最低的 epoch 停**（上一轮是 epoch 8，不要看到 20）
- 和现在的 `epoch_8_whole.pt` 做听评对比，谁顺留谁

### 3. 不要明天做的

- 不要训 flow / HiFT（数据量小，容易把官方声码器训坏）
- 不要在脏 ASR 上再开一轮 LLM
- 不要用 time-stretch 去「对齐语速」
- 本机 `transformers` 必须是 **4.51.3**，5.x 会出噪声

## 听评时注意

- Prompt 用干净的 3–6 秒原声，文本必须是这段 wav 的逐字稿。Spark 当时用的是 `vertin_01_00034`（约 6.2s）
- Zero-shot 前缀：`You are a helpful assistant.<|endofprompt|>` + prompt 转写
- SFT 是 bf16：`fp16=False`，加载后 `llm.float()`
- 故事推理：`python infer/infer.py -f infer/examples/story.txt -o story.wav`

## 若第二轮 LLM 已经明显像人话

再考虑要不要动 flow（后天以后的事）。现在还没到那一步。
