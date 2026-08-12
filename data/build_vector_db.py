import os
# 在一切导入之前，设置强制离线，防止任何网络请求
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HOME'] = r'F:\huggingface_cache'

import json
import chromadb
from chromadb.utils import embedding_functions

# 本地模型路径（你的快照路径）
LOCAL_MODEL_PATH = r"F:\huggingface_cache\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# 初始化嵌入函数（使用本地模型）
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=LOCAL_MODEL_PATH,
    device="cpu"  # 可指定设备
)

# 获取当前脚本所在目录 (data/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 向量数据库路径：data/chroma_db
db_path = os.path.join(BASE_DIR, "chroma_db")
client = chromadb.PersistentClient(path=db_path)

# 创建或获取集合
collection = client.get_or_create_collection(
    name="jobs",
    embedding_function=sentence_transformer_ef
)

# 读取 jobs.json（也在 data/ 目录下）
json_path = os.path.join(BASE_DIR, "jobs.json")
with open(json_path, "r", encoding="utf-8") as f:
    jobs = json.load(f)

ids = []
documents = []
metadatas = []
for i, job in enumerate(jobs):
    ids.append(str(i))
    doc = f"{job['title']} {job['company']} {job['description']} {' '.join(job['requirements'])}"
    documents.append(doc)
    metadatas.append({
        "title": job["title"],
        "company": job["company"],
        "location": job["location"],
        "link": job["link"],
        "category": job["category"]
    })

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)
print("向量数据库构建完成，职位数量：", collection.count())