from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents.tools import Tool
from langchain.prompts import PromptTemplate
import os
from tools.resume_parser import parse_resume
from tools.job_searcher import search_jobs
from tools.matcher import calculate_match

def create_agent():
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com/v1",
        temperature=0.7
    )

    tools = [parse_resume, search_jobs, calculate_match]

    # 定义 ReAct 提示模板（保持与原来相同的逻辑）
    template = """你是一位智能求职顾问，可以帮助用户分析简历、搜索匹配职位、评估匹配度。
你可以使用以下工具：
{tools}

工具名称：{tool_names}

请严格遵循以下格式回答：
Question: 用户的问题
Thought: 你应该怎么思考
Action: 要使用的工具名称，必须是 [{tool_names}] 中的一个
Action Input: 工具的输入
Observation: 工具返回的结果
... (可以重复 Thought/Action/Action Input/Observation 多次)
Thought: 我现在知道最终答案了
Final Answer: 用中文给出的最终回答

开始！
Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True), verbose=True)
    return agent_executor