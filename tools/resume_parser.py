from langchain.tools import tool
from langchain_openai import ChatOpenAI
import json
import os

# 初始化 DeepSeek LLM（用于提取结构化信息）
def get_deepseek_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com/v1",
        temperature=0
    )

@tool
def parse_resume(text: str) -> str:
    """解析用户输入的简历文本（或PDF提取后的文本），提取关键信息，返回JSON。"""
    llm = get_deepseek_llm()
    prompt = f"""从以下简历文本中提取关键信息，返回严格的JSON格式，不要添加任何解释。
JSON应包含字段：name (姓名), skills (技能列表), experience (工作经历列表，每项包含company, role, duration, details), education (教育背景列表，每项包含degree, institution, year), desired_role (期望职位)。

简历文本：
{text}
"""
    response = llm.invoke(prompt)
    # 简单清理可能的多余输出
    content = response.content.strip()
    if content.startswith("```json"):
        content = content[7:-3]
    elif content.startswith("```"):
        content = content[3:-3]
    return content.strip()