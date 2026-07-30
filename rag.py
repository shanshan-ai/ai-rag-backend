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

if __name__ == "__main__":
    #测试一下切分效果
    file_path = "test.txt"  

    #1.切分文档
    chunks = process_document(file_path)
    print(f"总共切分出了 {len(chunks)} 块文本 (chunks):\n - rag.py:40")

    #2.存入向量数据库
    vector_store = create_vector_store(chunks)
   
