import os
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import json

def get_deepseek_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com/v1",
        temperature=0
    )

@tool
def calculate_match(input_data: str) -> str:
    """详细对比简历摘要与职位详情，给出0-100匹配分和理由。
    输入必须是一个JSON字符串，包含 'resume_summary' 和 'job_details' 两个字段。"""
    try:
        data = json.loads(input_data)
        resume_summary = data["resume_summary"]
        job_details = data["job_details"]
    except (json.JSONDecodeError, KeyError):
        return "输入格式错误，请提供一个JSON字符串，包含 'resume_summary' 和 'job_details' 字段。"

    llm = get_deepseek_llm()
    prompt = f"""你是一位专业招聘顾问。请对比以下简历和职位要求，给出匹配度评分(0-100)及详细分析。

简历摘要：
{resume_summary}

职位详情：
{job_details}

输出格式：
评分：XX/100
匹配点：
- 点1
- 点2
不足之处：
- 不足1
建议：
- 建议1
"""
    response = llm.invoke(prompt)
    return response.content.strip()