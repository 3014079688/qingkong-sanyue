# tokenizer · 分词器卡片

实验记录：[`docs/experiments/20261123-中文BPE分词器自训练.md`](../../docs/experiments/20261123-中文BPE分词器自训练.md)

## 这两个文件是什么

自训的中文 BPE 分词器，Tokenizers 格式（`Tokenizer.from_file()` 直接加载）。**不是模型权重**，只决定文本怎么切块、怎么编号。

| 文件 | 方案 | 词表 | 体积 | 用途 |
| --- | --- | --- | --- | --- |
| `tokenizer_char.json` | 字符表起步 | 9,911 | ~520KB | 研究用语料匹配度，**有 OOV 缺陷** |
| `tokenizer_byte.json` | 字节起步 | 9,911 | ~740KB | 永不丢字，可安全用于任意文本 |

## 加载

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("models/tokenizer/tokenizer_char.json")
enc = tok.encode("秦川雄帝宅，函谷壯皇居。")
print(enc.tokens)          # ['秦', '川', '雄', '帝', '宅', '，', ...]
print(enc.ids)
print(tok.decode(enc.ids)) # 还原原文
```

## 训练数据

全唐诗第 0 卷（5,041 条 / 58,744 字符 / 3,353 个不同字符）。
`sha256 = 4f88f429938082523cdc72a1612bd7a7c3e3f7a33f8192c61ae51340efc3c92d`
原始数据不入库，获取方式见 `configs/tokenizer/bpe-20261123.json`。

## 已知限制（务必先读）

1. **领域极窄。** 语料只有唐诗。现代中文、技术术语、代码基本都在词表外。
2. **`tokenizer_char.json` 会丢字。** 词表外汉字输出 `<unk>`，解码后消失，且不报错。现代中文 105 字里丢 30 个。**不要用于任何生产场景。**
3. **不配套任何模型。** 该分词器的编号体系是自建的，与 Qwen / Llama / GPT 全部不兼容。要用于训练必须从随机初始化开始训。
4. **压缩率只在同域成立。** 唐诗上 0.71 token/字（优于 o200k 的 1.23），现代中文上立刻劣化到 1.25 并丢字。
