from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI RAG Knowledge Base Backend")
 #定义前端传过来的数据格式 （利用 FastAPI 的 Pydantic 检验）
class QueryRequest(BaseModel):
    question: str

#定义一个 POST 类型的聊天/检索窗口
@app.post("/api/chat")
def chat_with_rag(request: QueryRequest):
    #目前先用“模拟返回“，后续接入向量数据库和本地大模型
    user_question = request.question

    return {
        "code": 200,
        "message": "Success",
        "data": {
            "question": user_question,
            "answer": f"收到你的问题： 【{user_question}】。系统正在努力检索知识库中...(此为阶段一的模拟响应) "

        }
    }