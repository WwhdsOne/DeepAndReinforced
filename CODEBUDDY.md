# 项目概述

研究生入学前的深度学习算法学习与实验仓库。使用根目录统一的 `uv` Python 工程管理依赖和脚本入口。

# 语言约定

- 所有回复、解释、代码注释、commit message 使用中文
- 变量名、函数名保留英文

# 工具约定：RTK

所有读操作优先使用 `rtk`，替代 Claude Code 内置工具以节省 token：

| 内置工具 | RTK 替代 |
|---|---|
| `Grep`（内容搜索） | `rtk rg` |
| `Read`（读文件） | `rtk read` |
| `Bash` + `git` | `rtk git` |
| `Bash` + `gh` | `rtk gh` |
| `Bash` + `curl` | `rtk curl` |

仅当 RTK 无对应命令时（如 `Edit`、`Write`、`Agent`），才使用内置工具。

请用`rg`替代`grep`,以后不要再使用`grep`了。

# Python / uv 约定

- Python 3.11+，依赖由 `pyproject.toml` + `uv.lock` 统一管理
- 安装依赖：`uv sync --group dev`
- 运行脚本/测试：`uv run <command>`
- 测试：`uv run --group dev pytest -q`
- 新增实验放在 `experiments/<实验名>/`，内部维护 `src/`、`tests/`
- 新增命令行入口在根目录 `pyproject.toml` 的 `[project.scripts]` 注册
- 训练产物、下载数据、虚拟环境不提交 git

# 当前实验

## pic-classify — CIFAR-10 图片分类

基于 PyTorch + torchvision 的 CNN 分类器。

- 代码：`experiments/pic-classify/src/pic_classify/`
- 测试：`experiments/pic-classify/tests/`
- 数据集：`torchvision.datasets.CIFAR10(download=True)`，自动下载到 `experiments/pic-classify/data/`
- 模型产物：`experiments/pic-classify/artifacts/`

模型架构（`model.py`）：5 层卷积 → AdaptiveAvgPool → Flatten → Linear(128→10)

```bash
uv run pic-classify-train --epochs 5
uv run pic-classify-predict path/to/image.png --top-k 3
```

## neural_network — MNIST 分类

基于 NumPy 从零实现的全连接网络。代码：`experiments/neural_network/src/neural_network/`

```bash
uv run mnist-train
uv run mnist-predict path/to/image.png
uv run mnist-evaluate
uv run mnist-visualize
```

## fish-book — 鱼书学习实验

《深度学习入门：基于Python的理论与实现》章节代码，位于 `experiments/fish-book/`：
- `ch3/` — 神经网络基础（感知机、激活函数）
- `ch4/` — 梯度计算与优化
- `ch5/` — 神经网络层实现（乘法层、ReLU、Sigmoid、Softmax、Affine 等）
- `common/` — 通用工具函数

```bash
uv run python experiments/fish-book/ch5/multiLayer.py
```

# 学习日志

每日学习进度记录在 `learning-logs/`，目录结构如下：

```
learning-logs/
├── README.md          # 目录说明与日志内容建议
├── daily/             # 每日学习日志
│   └── YYYY-MM-DD.md  # 按日期命名，共 17 篇（2026-05-13 ~ 2026-06-02）
└── report/            # 阶段性总结报告
    └── YYYY-MM-DD_主题.md  # 跨日知识梳理、心得体会
```

## 每日日志内容规范

每天的学习成果以使用体验（项目内如何使用）为主，记录以下信息：

- **今日主题**：一句话概括当天学习方向
- **新增/改动文件**：表格列出所有变更文件及说明
- **核心知识点**：当天掌握的技术要点、API 用法、模型架构等
- **遇到的问题与解决方案**：踩坑记录、调试过程、根因分析
- **脚本入口**：可运行的命令（`uv run ...`）、测试命令
- **下一步计划**：待完成或待深入的方向

## 阶段性报告

当完成一个章节、一个课程、或一个实验模块后，在 `report/` 下撰写总结报告，包含：
- 一段时间内的学习进展概览
- 知识体系梳理（可配架构图、流程图）
- 关键经验教训
- 后续学习路线
