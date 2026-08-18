#向量存储和检索：文档向量生成(BGE-M3) -> 存入向量库(Milvus)-> 混合检索(稠密+稀疏向量) -> 结果去重、重排序 -> 返回文档

import sys
import torch.cuda
# BGE-M3 嵌入函数生成文档和查询的向量表示
from milvus_model.hybrid import BGEM3EmbeddingFunction
# Milvus客户端，数据类型，检索请求类，排序器
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from langchain_core.documents import Document
# 导入CrossEncoder重排序和NLI判断，优化检索结果的相关性排序
from sentence_transformers import CrossEncoder
# hashlib模块生成唯一 ID 的哈希值
import hashlib
from document_processor import *
from base import config
from base import logger

#路径配置,跨目录导入模块用
local_path = os.path.abspath(os.path.dirname(__file__))
rag_qa_path = os.path.abspath(os.path.dirname(local_path))
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

#存入向量数据库的关键操作，包括文档入库，混合检索等
class VectorStore:
    def __init__(self,
                 collection_name = config.MILVUS_COLLECTION_NAME,
                 host = config.MILVUS_HOST,
                 port = config.MILVUS_PORT,
                 database = config.MILVUS_DATABASE_NAME,
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database
        self.logger =logger
        #模型运行设备适应，优先GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # BGE-Reranker重排序模型初始化
        rerank_path = os.path.join(rag_qa_path,'models','bge-reranker-large')
        self.reranker = CrossEncoder(rerank_path.replace("\\", "\\\\"),device=self.device)
        #初始化BGE-M3模型，用于生成文档和查询的向量表示
        m3_path = os.path.join(rag_qa_path,'models','bge-m3')
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=m3_path,
            use_fp16=(self.device == "cuda"),  #GPU环境则启用半精度计算，提速并减少内存占用
            device=self.device,
        )

        #BGE-M3稠密向量输出固定为1024维
        self.dense_dim = self.embedding_function.dim['dense']
        #Milvus客户端初始化,链接url和数据库名称
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}" ,db_name = self.database)
        #(若不存在)创建所需集合


    def _create_collection(self):
        pass

    def add_documents(self,documents):
        pass

    def hybrid_search(self , query , source_fillter = None):
        pass

    def doc_rerank(self , k = config.RETRIEVAL_K):
        pass

    #子块列表依据父块id去重,然后返回去重后的父文档
    def unique_docs(self,sub_chunks):
        pass
    #将Milvus结果转换为LangChain的doc对象
    def _hit_to_doc(self,hit):
        pass







if __name__ == '__main__':
    v=VectorStore()