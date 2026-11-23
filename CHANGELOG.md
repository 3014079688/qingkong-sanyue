# CHANGELOG

本文件按时间倒序记录仓库的重大变动。日常实验记录放 `docs/experiments/`。

## [Unreleased]

### 2026-11-23

- 首个实验入库：中文 BPE 分词器自训练（`exp/20261123-token-bpe`）
  - `src/data/train_bpe.py`：从零训练 BPE，支持字符表 / 字节两种起始方案，强制校验语料 sha256
  - `configs/tokenizer/bpe-20261123.json`：全部超参、随机种子与语料校验值
  - `models/tokenizer/`：两个分词器产物 + 指标 + 切分明细 + 使用卡片
  - `docs/experiments/20261123-中文BPE分词器自训练.md`：正式实验记录
  - `docs/experiments/20261123-中文BPE分词器自训练-对话记录.md`：完整推导过程
  - `docs/pitfalls/20261123-分词器训练踩坑.md`：4 个踩坑记录
- 结论：自训分词器在诗词语料上压缩率 0.71 token/字（o200k 为 1.23），但现代中文 OOV 28.6% 且静默丢字
- 依赖首次补齐：`tokenizers`、`tiktoken`

### 2026-10-25

- 初始化仓库骨架：README、.gitignore、MIT LICENSE、文档模板与目录结构
- 确立留档规范：实验必须留下「输入 / 配置 / 结果 / 结论」四件套
