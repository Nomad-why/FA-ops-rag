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
        self._create_collection()



    def _create_collection(self):
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}#相似度较低也可能有关联，drop值过高可能导致召回率降低，需根据实际效果调整
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"集合 {self.collection_name}已存在无需加载")
        #加载到内存，可立即查询
        self.client.load_collection(self.collection_name)


    ##todo增量更新情况优化
    def add_documents(self,documents):
        """
        分块后的文档转换为向量并存储到Milvus
        :param documents:
        :return:
        """
        texts = [doc.page_content for doc in documents]

        embeddings = self.embedding_function(texts)
        data = []
        for i , doc in enumerate(documents):
            #将字符串转为字节流，然后md5哈希值作为文档的id，支持后续文档的数据去重和增量更新
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            #稀疏向量字典
            sparse_vector = {}
            # 获取第 i 行的稀疏向量数据
            row = embeddings['sparse'][[index], :]
            # 获取稀疏向量的非零值索引
            indices = row.indices
            # 获取稀疏向量的非零值
            values = row.data
            # 将索引和值配对，填充稀疏向量字典
            for idx, value in zip(indices, values):
                sparse_vector[idx] = value
            # 创建数据字典，包含所有字段
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings["dense"][i],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            })
            #检查是否有数据需要插入
        if data:
            # 使用 upsert 操作插入数据，覆盖重复 ID
            self.client.upsert(collection_name=self.collection_name, data=data)
            # 记录插入或更新的文档数量日志
            logger.info(f"已插入或更新 {len(data)} 个文档")


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

    andir='../data/ai_data'
    documents = process_documents(andir)
    vector_store.add_documents(documents)
