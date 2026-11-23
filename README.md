# qingkong-sanyue · 自研 AI 留档

自研 AI 项目的长期留档仓库：代码、实验记录、数据集说明、模型权重索引与阶段性总结统一归档在这里。

> 留档原则：**能复现 > 好看**。每次实验都应留下「输入 / 配置 / 结果 / 结论」四件套。

## 目录结构

```
qingkong-sanyue/
├── ai-assistant/       # 已归档项目：本地私人 AI 助手（见下方说明）
├── bpe-demo/           # 已归档项目：中文 BPE 分词器（见下方说明）
├── configs/            # 训练与实验配置（yaml/json），一切可复现的前提
├── data/
│   ├── raw/            # 原始数据（不入库，见 .gitignore）
│   └── processed/      # 清洗后的中间数据（不入库）
├── docs/               # 设计文档、阶段总结、踩坑记录
│   └── 实验记录模板.md   # 每次实验抄一份，填完再归档
├── models/             # 模型权重索引与卡片（大权重走外部存储）
├── notebooks/          # 探索性分析、可视化、临时验证
├── src/                # 正式代码（可复用、可测试）
│   ├── data/           #   数据加载与预处理
│   ├── models/         #   模型定义
│   ├── training/       #   训练与评测循环
│   └── utils/          #   工具函数
└── CHANGELOG.md        # 按时间倒序的大事记
```

## 已归档项目

独立成型的项目以子目录形式并入，各自**保留完整的提交历史**（`git subtree`），不套用上面的 `src/` 骨架。可复现的细节写在各自的 README 里。

| 目录 | 项目 | 说明 | 入口 |
|---|---|---|---|
| `ai-assistant/` | 本地私人 AI 助手 | Ollama + FastAPI + 单文件界面，带本地知识库检索；全程离线，数据不出本机 | [BUILD-LOG.md](ai-assistant/BUILD-LOG.md) |
| `bpe-demo/` | 中文 BPE 分词器 | 从零训练分词器，与商用分词器对比 token 切分效果 | [token 对话记录](bpe-demo/docs/2026-11-23-token对话记录.md) |

新增项目沿用同样方式并入：

```bash
git subtree add --prefix=<项目名> <项目本地路径> main -m "feat: 并入 <项目名>"
```

## 快速约定

- **分支**：`main` 只放能跑通的东西；新实验走 `exp/<日期>-<主题>` 分支。
- **提交**：一句话说清改动，`feat / fix / exp / docs / refactor` 前缀任选其一。
- **实验**：默认记录在 `docs/experiments/YYYYMMDD-<主题>.md`，格式照抄模板。
- **数据**：原始数据一律不进 Git，只在 `docs/` 里写明来源、获取方式与校验值。
- **随机性**：所有实验固定随机种子，并在配置文件里显式写出。

## 复现步骤

```bash
git clone https://github.com/3014079688/qingkong-sanyue.git
cd qingkong-sanyue
python -m venv .venv && ./.venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

`requirements.txt` 在补齐具体依赖后加入。

## License

MIT © 2026 梁泽斌
