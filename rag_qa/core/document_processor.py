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
def load_document_from_directory(directory_path):
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
                        doc.metadata["timestamp"] = datetime.now()
                    documents.extend(loaded_docs)
                    logger.info(f"成功加载{file_path}路径下文件")
                except Exception as e:
                    logger.error(f"{file_path}路径文件加载异常：{e}")
            else:
                logger.error(f"不支持的文件加载类型")
    return documents






if __name__ == "__main__":
    directory_path = '../data/ai_data'
    documents= load_document_from_directory(directory_path)
    print(f'文档内容为：{documents[0] if documents else "无内容"}')