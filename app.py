import streamlit as st
import requests

# 页面基本配置
st.set_page_config(page_title="AI RAG 知识库问答", page_icon="🤖", layout="centered")

st.title("🤖 AI 智能知识库问答系统")
st.write("基于 FastAPI + ChromaDB + 智谱 GLM 的 RAG 系统")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 展示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户的输入
if prompt := st.chat_input("请输入你想问关于知识库的问题..."):
    # 将用户输入追加到历史记录并展示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用我们写好的 FastAPI 后端接口
    with st.chat_message("assistant"):
        with st.spinner("AI 正在努力检索并思考..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/api/chat", 
                    json={"question": prompt},
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # 假设后端返回的字段里包含答案，比如 result["answer"]
                    answer = result.get("data", {}).get("answer", "后端返回格式不匹配，未找到 answer 字段。")
                else:
                    answer = f"请求后端失败，状态码: {response.status_code}"
            except Exception as e:
                answer = f"连接后端出错，请检查 FastAPI 是否正常运行。错误信息: {str(e)}"
            
            st.markdown(answer)
            # 将 AI 的回复也追加到历史记录
            st.session_state.messages.append({"role": "assistant", "content": answer})
