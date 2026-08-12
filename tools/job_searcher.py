import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HOME'] = r'F:\huggingface_cache'

from langchain.tools import tool
import chromadb
from chromadb.utils import embedding_functions
import json

# 本地模型路径
LOCAL_MODEL_PATH = r"F:\huggingface_cache\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=LOCAL_MODEL_PATH,
    device="cpu"
)

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(TOOLS_DIR, "..", "data", "chroma_db")

@tool
def search_jobs(profile_json_str: str) -> str:
    """根据用户画像JSON，在职位库中语义搜索最匹配的5个职位。"""
    try:
        profile = json.loads(profile_json_str)
    except json.JSONDecodeError:
        return "输入的画像格式无效，请确保是有效的JSON。"
    
    query_parts = []
    # 技能
    if "skills" in profile and profile["skills"]:
        query_parts.extend([str(s) for s in profile["skills"]])
    # 期望职位（兼容多种字段名）
    desired = profile.get("desired_role") or profile.get("target_position")
    if desired:
        query_parts.append(str(desired))
    # 工作经历
    if "experience" in profile:
        for exp in profile["experience"]:
            if "role" in exp:
                query_parts.append(str(exp["role"]))
            if "details" in exp:
                details = exp["details"]
                if isinstance(details, list):
                    query_parts.extend([str(d) for d in details])
                else:
                    query_parts.append(str(details))
    
    query_text = " ".join(query_parts) if query_parts else profile_json_str

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection("jobs", embedding_function=sentence_transformer_ef)
    
    results = collection.query(query_texts=[query_text], n_results=5)
    jobs_output = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        jobs_output.append(
            f"岗位 {i+1}: {meta['title']} @ {meta['company']} ({meta['location']})\n"
            f"匹配相似度: {1-dist:.2f}\n"
            f"职位链接: {meta['link']}\n"
            f"简介: {doc[:150]}..."
        )
    return "\n\n".join(jobs_output) if jobs_output else "没有找到匹配的职位。"