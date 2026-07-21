import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import httpx
from huggingface_hub import HfApi, get_session

# 同样增加超时
session = get_session()
session.timeout = httpx.Timeout(120.0, connect=30.0)

api = HfApi()
try:
    info = api.dataset_info("uobinxiao/open_tables_icttd_for_table_detection")
    print("✅ 成功获取仓库信息！")
    print("文件数量:", len(info.siblings))
except Exception as e:
    print("❌ 失败:", e)