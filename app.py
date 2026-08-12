import gradio as gr
from agent import create_agent

agent = create_agent()

def chat(message, history):
    response = agent.invoke({"input": message})  # 新版本用 invoke
    return response["output"]  # 返回最终答案字符串

demo = gr.ChatInterface(
    fn=chat,
    title="🤖 多领域智能求职匹配 Agent",
    description="描述您的技能或粘贴简历，我将为您推荐最匹配的职位",
    examples=[...]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)