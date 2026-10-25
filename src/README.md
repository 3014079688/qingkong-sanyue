# src 目录约定

正式代码放这里，要求是**能被 import、能复现**，一次性脚本请放 `notebooks/`。

| 子目录 | 职责 |
| --- | --- |
| `data/` | 数据加载、清洗、特征工程；对外暴露 Dataset / DataLoader |
| `models/` | 模型定义与结构变体；每个模型配一个同名配置 |
| `training/` | 训练循环、评测、指标计算 |
| `utils/` | 通用工具：日志、随机种子、IO、可视化 |

规则：

- 路径、超参不写死在代码里，一律从 `configs/` 传入。
- 随机种子由 `utils.set_seed` 统一设置，禁止在业务代码里直接调 `torch.manual_seed`。
- 任何临时调试代码不允许合入 `main`。
