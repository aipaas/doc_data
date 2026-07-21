from huggingface_hub import HfApi

# 1. 创建HfApi实例，指定镜像站和token
api = HfApi(endpoint="https://hf-mirror.com", token="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")

repo_id="ServiceNow/BigDocs-7.5M"

# 2. 获取数据集所有文件路径
files = api.list_repo_files(
    repo_id=repo_id,
    repo_type="dataset"  # 必须指定为dataset
)

# 3. 生成完整的下载URL并保存，按子路径（目录）分组输出，同一子路径的 url 会排在一起
from collections import defaultdict

base_url = "https://hf-mirror.com"

# 将文件按目录分组
groups = defaultdict(list)
for file in files:
    if "/" in file:
        dirpath = file.rsplit("/", 1)[0]
    else:
        dirpath = ""  # 根目录
    groups[dirpath].append(file)

with open("download_urls.txt", "w", encoding="utf-8") as f:
    # 按目录排序，目录内按文件名排序，保证同一子路径的 url 连续输出
    for dirpath in sorted(groups.keys()):
        for file in sorted(groups[dirpath]):
            f.write(f"{base_url}/{repo_id}/{file}\n")

total = sum(len(v) for v in groups.values())
print(f"✅ 成功导出 {total} 个文件的下载链接到 download_urls.txt（按子路径分组）")