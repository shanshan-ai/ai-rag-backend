from fastapi import FastAPI
from pydantic import BaseModel
from rag import search_knowledge_base,generate_answer

app = FastAPI(title="AI RAG Knowledge Base Backend")

 #定义前端传过来的数据格式 （利用 FastAPI 的 Pydantic 检验）
class QueryRequest(BaseModel):
    question: str
    k: int = 2

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI RAG Knowledge Base Backend!"}

#定义一个 POST 类型的聊天/检索窗口
@app.post("/api/chat")
def chat_with_rag(request: QueryRequest):
    user_question = request.question

    try:
        #调用rag.py中的真实检索函数
        results = search_knowledge_base(user_question, k=request.k)

        #调用 rag.py 中大模型生成函数，让 GLM-5.2 结合检索到的资料回答
        ai_answer = generate_answer(user_question, results)

        #提取检索到的文本内容
        retrieved_texts = [doc.page_content for doc in results]

        return {
            "code": 200,
            "message": "Success",
            "data": {
                "question": user_question,
                "retrieved_docs": retrieved_texts,
                "answer": ai_answer
            }
        }
    except Exception as e:
        return {
            "code": 500,
            "message": str(e),
            "data": None
        }
