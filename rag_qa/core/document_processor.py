# core/document_processor.py
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownTextSplitter
from datetime import datetime

from nltk import data
from pypika.clickhouse import dates_and_times

from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter
from rag_qa.edu_text_spliter import AliTextSplitter
from rag_qa.edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
from base import config , logger

# 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}
##加载模块,从指定目录加载所有文件,以递归形式进行,返回文档列表
def load_documents_from_directory(directory_path):
    """
    加载指定目录中所有支持类型的文件，并为每个文档添加原数据
    :param directory_path: 文档所在目录的绝对路径
    :return: 文档列表，LangChain Document 对象
    """
    documents = []
    #拿到所有支持加载的文件类型
    supported_extensions = list(document_loaders.keys())
    #从目录名提取学科类别，例如fa_data->fa作为source
    source = os.path.basename(directory_path).replace("_data", "")
    #便利目录下每个文件，逐一处理
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file_path)[1].lower()
            # print(f'{file_extension}')
            ##根据后缀创建加载器，如果是txt文件，使用utf-8编码
            if file_extension in supported_extensions:
                try:
                    loaders = document_loaders[file_extension]
                    if file_extension == ".txt":
                        loader = loaders(file_path, encoding='utf-8')
                    else:
                        loader = loaders(file_path)
                    ##加载文档
                    loaded_docs = loader.load()
                    ##加载后给每个文档添加元数据，用于RAG后续的检索和溯源
                    for doc in loaded_docs:
                        # 文档类别、文件路径、添加时的时间戳
                        doc.metadata["source"] = source
                        doc.metadata["file_path"] = file_path
                        doc.metadata["timestamp"] = datetime.now().isoformat()
                    documents.extend(loaded_docs)
                    logger.info(f"成功加载{file_path}路径下文件")
                except Exception as e:
                    logger.error(f"{file_path}路径文件加载异常：{e}")
            else:
                logger.error(f"不支持的文件加载类型")
    return documents


# 处理文档并进行分层切分，返回子块结果
def process_documents(directory_path, parent_chunk_size=config.PARENT_CHUNK_SIZE,
                     child_chunk_size=config.CHILD_CHUNK_SIZE,
                     chunk_overlap=config.CHUNK_OVERLAP):
    # 从指定目录加载所有文档
    documents = load_documents_from_directory(directory_path)
    # 记录加载的文档总数日志
    logger.info(f"加载的文档数量: {len(documents)}")

    # 初始化父块和子块分词器（通用）
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # 初始化 Markdown 专用分词器
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    # 初始化空列表，用于存储所有子块
    child_chunks = []
    # 遍历每个原始文档，带上索引 i
    for i, doc in enumerate(documents):
        # 获取文件扩展名
        file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower()

        # 选择切分器
        is_markdown = (file_extension == ".md")
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
        logger.info(f"处理文档: {doc.metadata['file_path']}, 使用切分器: {'Markdown' if is_markdown else 'ChineseRecursive'}")

        # 使用父块分词器将文档切分为父块
        parent_docs = parent_splitter_to_use.split_documents([doc])
        # 遍历每个父块，带上索引 j
        for j, parent_doc in enumerate(parent_docs):
            # 为父块生成唯一 ID，格式为 "doc_i_parent_j"
            parent_id = f"doc_{i}_parent_{j}"
            # 将父块 ID 添加到元数据
            parent_doc.metadata["parent_id"] = parent_id
            # 将父块内容存储到元数据
            parent_doc.metadata["parent_content"] = parent_doc.page_content

            # 使用子块分词器将父块切分为子块
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])
            # 遍历每个子块，带上索引 k
            for k, sub_chunk in enumerate(sub_chunks):
                # 为子块添加父块 ID 到元数据
                sub_chunk.metadata["parent_id"] = parent_id
                # todo此处为子块添加父块内容到元数据，会导致内存爆炸，后续mysql——qa和rag——qa联调时做修改，子块元数据仅存储父块id，父块的内容存入数据库，仅用于子块匹配后溯源
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                # 为子块生成唯一 ID，格式为 "parent_id_child_k"
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
                # 将子块添加到子块列表中
                child_chunks.append(sub_chunk)

    logger.info(f"子块数量: {len(child_chunks)}")
    return child_chunks


if __name__ == '__main__':

    chunks = process_documents(
        r'C:\Users\10413\PycharmProjects\FA-ops-rag\rag_qa\data\ai_data',
        config.PARENT_CHUNK_SIZE,
        config.CHILD_CHUNK_SIZE,
        config.CHUNK_OVERLAP,
    )
    print(chunks)


