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
import numpy as np

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
            row = embeddings['sparse'][[i]]
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
                "dense_vector": np.array(embeddings["dense"][i], dtype=np.float32),
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


    def hybrid_search(self , query , k = config.RETRIEVAL_K, source_filter = None):
        # 使用 BGE-M3 嵌入函数生成查询的嵌入
        query_embeddings = self.embedding_function([str(query)])
        dense_query_vector = np.array(query_embeddings["dense"][0], dtype=np.float32)
        sparse_query_vector = {}
        row = query_embeddings["sparse"][[0]]
        # 获取稀疏向量的非零值 index 和 value 并配对，写入稀疏向量字典
        indices = row.indices
        values = row.data
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value

        # 初始化过滤表达式，默认不过滤
        filter_expr = f"source == '{source_filter}'" if source_filter else ""

        # 稠密向量搜索请求
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )

        # 稀疏向量搜索请求
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )

        # 创建加权排序器，稀疏向量权重 0.7，稠密向量权重 1.0
        ranker = WeightedRanker(1.0, 0.7)
        # 执行混合搜索，返回 Top-K 结果
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
        )[0]

        # 将搜索结果转换为 Document 对象列表
        sub_chunks = [self._hit_to_doc(hit["entity"]) for hit in results]
        print(f'sub_chunks-->{len(sub_chunks)}')
        # 从子块中提取去重的父文档
        parent_docs = self.unique_docs(sub_chunks)
        # 如果只有1个文档，直接返回跳过重排序
        if len(parent_docs) < 2:
            return parent_docs[:config.CANDIDATE_M]
            # 如果有父文档，进行重排序
        if parent_docs:
            ranked_parent_docs = self.doc_rerank(query,parent_docs)
        else:
            ranked_parent_docs = []
        # 返回前 k 个重排序后的文档
        return ranked_parent_docs[:config.CANDIDATE_M]

    ##BGE-eranker重排序
    def doc_rerank(self ,query , parent_docs):
        # 创建查询与文档内容的配对列表
        pairs = [[query, doc.page_content] for doc in parent_docs]
        # 使用 BGE-Reranker 计算每个配对的得分
        scores = self.reranker.predict(pairs)
        # 根据得分从高到低排序文档
        ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]

        return ranked_parent_docs


    # 如果没有父文档，返回空列表

    #子块列表依据父块id去重,然后返回去重后的父文档
    def unique_docs(self,sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
        # 返回去重后的父文档列表
        return unique_docs


    #将Milvus结果转换为LangChain的doc对象
    def _hit_to_doc(self,hit):
        # 创建并返回 Document 对象，填充内容和元数据
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )







if __name__ == '__main__':
    v=VectorStore()
    andir='../data/ai_data'
    documents = process_documents(andir)
    v.add_documents(documents)
    print(v.hybrid_search("Today is a nice day"))