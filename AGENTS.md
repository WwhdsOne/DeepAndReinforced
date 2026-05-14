# 项目说明

本仓库用于研究生入学前的算法学习与相关实验。

当前仓库使用根目录统一的 `uv` Python 工程进行依赖和脚本管理。后续新增的 Python 程序、实验代码和可执行入口，都应尽量接入根目录 `pyproject.toml`，避免每个子目录重复维护虚拟环境和依赖。

# 仓库约定

## Python / uv

- 使用仓库根目录的 `pyproject.toml` 和 `uv.lock` 统一管理依赖。
- 使用 `uv sync --group dev` 安装开发依赖。
- 使用 `uv run <command>` 执行脚本、测试和工具。
- 每个算法实验单独放在 `experiments/<experiment-name>/` 下。
- 每个实验目录内部可以有自己的 `src/`、`tests/`、`README.md`。
- 新增 Python 包时，优先放在对应实验目录的 `src/` 下，保持实验边界清晰。
- 训练产物、下载数据和虚拟环境不提交到 git。

## 当前程序：`pic-classify`

- `pic-classify` 是当前仓库中的图片分类实验程序。
- 代码位置：`experiments/pic-classify/src/pic_classify/`
- 测试位置：`experiments/pic-classify/tests/`
- 训练入口：`uv run pic-classify-train`
- 预测入口：`uv run pic-classify-predict`
- 数据集来源：`torchvision.datasets.CIFAR10(download=True)`，不再依赖旧的本地 `cifar-10-batches-py` 目录。
- 数据默认下载到 `experiments/pic-classify/data/`
- 模型产物默认输出到 `experiments/pic-classify/artifacts/`

### 常用命令

```bash
uv sync --group dev
uv run --group dev pytest -q
uv run pic-classify-train --epochs 5
uv run pic-classify-predict path/to/image.png
```

# 工具约定

## RTK - Rust Token Killer

所有命令前缀 `rtk`，替代 Claude Code 内置的同类工具以节省 token。

### 优先级规则

RTK 命令优先于 Claude Code 内置工具，功能重叠时优先使用 RTK：

| Claude Code 内置工具 | RTK 替代命令 | 说明 |
|---|---|---|
| `Glob`（文件搜索） | `rtk find` / `rtk tree` / `rtk ls` | 压缩目录输出 |
| `Grep`（内容搜索） | `rtk grep` | 按文件分组、截断、去空白 |
| `Read`（读文件） | `rtk read` | 智能过滤，省去无用行 |
| `Bash` + `git` | `rtk git` | 紧凑 git 输出 |
| `Bash` + `gh` | `rtk gh` | 紧凑 GitHub CLI 输出 |
| `Bash` + `curl` | `rtk curl` | 自动检测 JSON，schema-only 模式 |
| `Bash` + `diff` | `rtk diff` | 仅显示变更行 |

仅当 RTK 无对应命令时（如 `Edit`、`Write`、`Agent` 等写操作和复杂操作），才使用内置工具。

### 常用命令

#### Node.js / Frontend

```bash
rtk pnpm install / add / run build
rtk npm run <script>
rtk npx tsc / eslint / prisma
rtk vitest run
rtk next build
rtk lint
rtk prettier --check .
rtk playwright test
rtk tsc --noEmit
```

#### Python

```bash
rtk pytest
rtk ruff check / format
rtk mypy .
rtk pip install / list
```

#### Rust

```bash
rtk cargo build / test / clippy / fmt
```

#### Go

```bash
rtk go build / test / vet
rtk golangci-lint run
```

#### .NET / Ruby

```bash
rtk dotnet build / test
rtk rspec / rake / rubocop
```

#### Infrastructure

```bash
rtk aws <service> <command>
rtk docker ps / logs / compose
rtk kubectl get / describe / logs
rtk psql <query>
```

#### Meta Commands

```bash
rtk gain
rtk gain --history
rtk discover
```

# 语言约定

- 所有回复、解释、代码注释、commit message 均使用中文。
- 变量名、函数名等遵循项目约定，可保留英文。
- 代码注释使用中文，除非项目已有英文注释惯例。
- 错误信息和日志的解读用中文说明。
