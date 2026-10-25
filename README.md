# qingkong-sanyue · 自研 AI 留档

自研 AI 项目的长期留档仓库：代码、实验记录、数据集说明、模型权重索引与阶段性总结统一归档在这里。

> 留档原则：**能复现 > 好看**。每次实验都应留下「输入 / 配置 / 结果 / 结论」四件套。

## 目录结构

```
qingkong-sanyue/
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
