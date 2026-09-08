# 微调

| 路径 | 说明 |
| --- | --- |
| `ckpts/epoch_8_whole.pt` | 推荐。验证集 loss 最低（4.809） |
| `ckpts/epoch_7_whole.pt` | 几乎一样 |
| `ckpts/epoch_10_whole.pt` | 仍接近最好 |
| `data/` | raw_wav、processed clips、source 视频 |
| `scripts/` | Windows 数据准备 |
| `spark/` | Spark 上装环境、打 parquet、训 LLM |
| `conf/cosyvoice3_sft.yaml` | SFT 配置 |

只替换 LLM。flow / hift 继续用 `runtime` 里的官方底模。

SFT 权重几乎全是 bf16。推理时不要 `--fp16`，加载后 `model.llm.float()`。
