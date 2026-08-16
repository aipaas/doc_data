# Document OCR Data Toolkit

本仓库保存 OCR/文档数据源目录、下载与核验脚本、项目规范，以及两个可独立安装的 Codex skill。根 README 是仓库的人类使用入口；skill 被复制到其他环境后不读取本 README、仓库文档或其他兄弟文件。

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| `skills/` | 可单独复制使用的数据筛选与资产下载 skill |
| `scripts/` | AI Studio、ModelScope、Hugging Face 和客服附件工具 |
| `data_sources/` | 平台目录快照、决策报告、缓存和来源清单 |
| `docs/` | 项目核验规范与标题修复采集进度快照 |
| `test/` | 下载器、完整性检查和网络行为测试 |

`data_sources` 中的报告和缓存是特定日期的证据快照，不是数据集 payload，也不自动代表样本或完整资产已核验。

## 脚本环境

脚本使用 Python 3.10+。仓库当前没有统一 lockfile，按实际入口安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install requests httpx huggingface_hub tqdm psutil
```

ModelScope 完整下载/审计额外需要官方 `modelscope` 包；AI Studio Git 仓库下载额外需要官方 `aistudio` CLI/SDK；运行测试需要 `pytest`。平台 SDK 的版本和登录方式应按其当前官方文档选择，本仓库不固定个人环境版本。

按需通过环境变量或被 `.gitignore` 排除的 `.env` 提供凭据：

```text
HF_TOKEN=...
MODELSCOPE_API_TOKEN=...
AISTUDIO_ACCESS_TOKEN=...
```

公开数据通常不需要 token。不要把真实凭据写入脚本、Markdown、TSV、日志或 Git；历史中出现过的凭据必须在平台侧撤销或轮换，仅从新版本删除并不能使旧凭据失效。

## 使用脚本

先运行目录收集、`--list-only`、`--dry-run` 或只读审计，再启动完整下载。ModelScope 脚本带有历史默认盘符，非原始环境必须显式传入 `--root`。

```bash
# 收集/查看平台目录，不下载 payload
python scripts/aistudio_ocr_catalog.py --refresh --collect-only
python scripts/modelscope_ocr_catalog.py --refresh --count 20

# 只列 AI Studio 下载计划；不下载 payload
python scripts/aistudio_batch_downloader.py \
  --output-dir /mnt/data/aistudio --list-only --print-commands

# ModelScope 空间与候选预检
python scripts/modelscope_batch_downloader.py \
  --root /mnt/data/modelscope --dry-run

# Hugging Face 镜像完整下载与核验
python scripts/hf_mirror_downloader.py owner/dataset \
  --data-root /mnt/data --revision main

# 导出固定 revision 的镜像 URL，不下载 payload
python scripts/list_urls.py owner/dataset \
  --revision main --output download_urls.txt

# 从指定本地工作簿和挂载盘生成资产表；--volume 可重复
python scripts/aistudio_asset_table.py \
  --workbook /path/to/文档解析数据集核心资产统计.xlsx \
  --volume /Volumes/aistudio1 \
  --volume /Volumes/shareData6 \
  --output /tmp/aistudio资产表_全字段.tsv

# 默认只写相对挂载点的存储路径；本机排查时才显式添加：
# --absolute-storage-paths

# 对已有 ModelScope 目录做只读审计
python scripts/modelscope_download_audit.py \
  --root /mnt/data/modelscope --json
```

移除 `--list-only` 或 `--dry-run` 前，先确认固定 revision、预计完整大小、单盘空间、凭据和目标目录。`scripts/download/` 是客服附件项目工具，其中部分命令会原地更新 CSV；运行前必须查看 `--help` 并明确输入、输出和备份边界。

## Skills

### `ocr-title-screening`

批量筛选 OCR/文档数据集，输出具体输入与标签、数据类型、可用性、物理多页状态、标题修正关联程度和证据说明。

最低输入：一份 XLSX、CSV、TSV 或 JSON 资产表，或一组明确的数据集名称/仓库 ID；每条记录应有稳定行键或精确数据集名。输出路径可省略并使用 skill 内默认值。既有 checkpoint、本地路径、字段映射、项目枚举、任务细化和镜像策略均为可选输入。

### `ocr-data-assets`

执行完整下载、断点续传、完整性核验、未下载字段复核和原子更新资产 TSV，并把完整本地资产与未下载字段修正分开交付。

最低输入：资产表或明确的数据集名称/仓库 ID；完整下载时还需提供已挂载文件系统上的目标目录。输出路径、字段映射、空间余量、镜像策略和既有完成证据均可选；未指定安全余量时默认保留 20 GiB。

安装到 Codex：

```bash
mkdir -p ~/.codex/skills
cp -R skills/ocr-title-screening ~/.codex/skills/
cp -R skills/ocr-data-assets ~/.codex/skills/
```

调用示例：

```text
Use $ocr-title-screening to review ./inventory.xlsx and write ./screening.xlsx.
Use $ocr-data-assets to download selected rows from ./inventory.tsv into /mnt/data.
```

自定义资产表只需提供稳定数据集标识；字段名含义不明确时，同时提供字段映射和受控枚举。源资产表默认只读，只有用户明确授权时才允许回写。

## 核心规则

- 先复用资产表、checkpoint 和本地证据，再查官方元数据，最后才做受控采样；不重复下载或核验。
- 区分复用、官方元数据、代表样本和完整本地资产，不能把较弱证据写成较强证据。
- 镜像只负责传输，来源始终记录官方地址；有真实配对时写`输入 -> 标签`，无标签时只写具体输入。
- 获取状态与任务可用性分开；容量不足不等于`暂时不可用`，重叠类别按`仅评测 > 缺label > 需处理`处理。
- 多页只认物理多页文档或可重建的有序页面关系。
- 完整资产必须有固定发布边界、单盘空间、完整清单和可读性证据；样本、partial 和 metadata 不进入完整下载 TSV。
- 一个数据集一行，小批量原子 checkpoint；只清理本轮临时文件，不触碰既有资产或其他任务目录。

仓库版和本地工作版允许为可移植性调整措辞、路径和默认输出名，但以上规则、证据边界和判定优先级必须保持一致。

## 文档与测试

- [文档 OCR 数据集筛选、下载与核验标准](docs/文档OCR数据集筛选评测与核验标准.md)：项目 16 字段、证据层级、下载完整性、多页和标题关联口径。
- [文档 OCR 标题修复数据采集进度](docs/文档OCR标题修复_完整下载分析与执行队列_20260729.md)：基于 2026-07-30 私有资产表快照的进度、价值评估和后续队列；原 XLSX 与包含本机路径的原始 TSV 不在仓库中，可提交的快照应使用生成脚本的默认相对存储路径。

安装测试依赖后运行：

```bash
PYTHONPATH=scripts python -m pytest
```

Windows PowerShell 可先执行 `$env:PYTHONPATH = "scripts"`。两个 skill 还应通过 Codex `skill-creator` 的结构校验。项目文档和脚本可辅助人工维护，但都不是 skill 的运行依赖。
