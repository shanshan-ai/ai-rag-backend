from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
#向量化组件以及Chorm向量数据库
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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
    print("正在加载轻量级本地嵌入模型(SentenceTransformers)... - rag.py:20")
    # 使用开源的轻量级中文嵌入模型，它会在本地自动下载模型文件，首次运行可能需要一些时间
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("正在构建并持久化Chroma向量数据库中... - rag.py:24")
    #将文本块通过嵌入模型转化为向量，并保存在本地的"./chroma_db"文件夹中
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("向量数据库构建完成! - rag.py:31")
    return vector_store

def search_knowledge_base(query: str,k: int = 2):
    """根据用户问题，从向量数据库中检索最相关的文本块"""
    print(f"\n正在加载嵌入模型并连接向量数据库以检索问题： ‘{query} - rag.py:36")   

    #1.初始化相同的嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #2.从本地文件夹加载持久化的 Chroma 数据库
    vector_store = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )

    #3.相似度检索(Similarity Search)
    results = vector_store.similarity_search(query, k=k)

    print(f"\n检索结果(Top {k}) - rag.py:50")
    for i,doc in enumerate(results):
        print(f"[{i+1}] 内容: {doc.page_content} - rag.py:52")
    return results

if __name__ == "__main__":
    file_path = "test.txt"  

    #1.切分文档并存入向量数据库
    chunks = process_document(file_path)
    vector_store = create_vector_store(chunks)

    #2.测试检索功能
    test_query = "什么是人工智能？"
    search_knowledge_base(test_query, k=2)
                                   