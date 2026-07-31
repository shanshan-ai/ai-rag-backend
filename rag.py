import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
#向量化组件以及Chorm向量数据库
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()#全局初始化配置 加载 .env 文件中的 API Key

#初始化智谱大模型(GLM-5.2)
llm = ChatOpenAI(
    model="glm-5.2",
    temperature=0.3,
    max_tokens=1024,
)

#定义大模型提示词模板
template = """你是一个智能知识库问答助手。请根据以下提供的参考资料，简洁、准确地回答用户的问题。如果参考资料中没有相关信息，请直接回答"抱歉，根据本地知识库未能找到相关内容。

参考资料:
{context}
用户问题:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

#==========================文档处理与向量库构造函数==========================
def process_document(file_path:str):
    # 加载本地文件并进行切分
    loader = TextLoader(file_path,encoding="utf-8")
    document = loader.load()

    # 文本切块：每块200字符，重叠20字符
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(document)

    return chunks

def create_vector_store(chunks):
    """将文本块向量化并存储到Chroma向量数据库中"""
    print("正在加载轻量级本地嵌入模型(SentenceTransformers)... - rag.py:44")
    # 使用开源的轻量级中文嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("正在构建并持久化Chroma向量数据库中... - rag.py:48")
    #将文本块通过嵌入模型转化为向量，并保存在本地的"./chroma_db"文件夹中
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("向量数据库构建完成! - rag.py:55")
    return vector_store

#==========================知识库检索函数==========================
def search_knowledge_base(query: str,k: int = 2):
    """根据用户问题，从向量数据库中检索最相关的文本块"""
    print(f"\n正在加载嵌入模型并连接向量数据库以检索问题： ‘{query} - rag.py:61")   

    #1.初始化相同的嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #2.从本地文件夹加载持久化的 Chroma 数据库
    vector_store = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )

    #3.相似度检索(Similarity Search)
    results = vector_store.similarity_search(query, k=k)

    print(f"\n检索结果(Top {k}) - rag.py:75")
    for i,doc in enumerate(results):
        print(f"[{i+1}] 内容: {doc.page_content} - rag.py:77")
    return results

#==========================大模型生成函数==========================
def generate_answer(question: str,retrieved_docs: list) -> str:
    """根据用户问题和检索到的文本块，调用大模型生成答案"""
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    formatted_prompt = prompt.format(context=context_text, question=question)
    response = llm.invoke(formatted_prompt)
    return response.content

   
if __name__ == "__main__":
    file_path = "test.txt"  

    #1.切分文档并存入向量数据库
    #chunks = process_document(file_path)
    #vector_store = create_vector_store(chunks)

    #2.测试检索功能
    test_query = "流光似水中的主要人物为哪些？"
    docs = search_knowledge_base(test_query, k=2)
    answer = generate_answer(test_query, docs)
    print("\n大模型最终回答： - rag.py:100", answer)
                                   