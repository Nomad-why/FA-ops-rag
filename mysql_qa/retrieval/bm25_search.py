# 基于BM25+Softmax归一化的文本检索模块，包含问题加载，分词，BM25评分。Softmax归一化。在向量检索之前，解决向量检索可能对关键字类搜索匹配能力差的问题。
# BM25
from rank_bm25 import BM25Okapi
import numpy as np
from mysql_qa.utils.preprocess import preprocess_text
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import RedisClient
from base import logger , config


#BM25Serch Class
class BM25Search:
    def __init__(self,mysql_client,redis_client):
        self.logger = logger
        self.mysql_client = mysql_client
        self.redis_client = redis_client
        self.bm25= None
        #分词后的问题列表
        self.questions =None
        #初始化原始问题列表
        self.original_questions =None
        self._loda_data()


    #todo 案例数据有更新时如何实现bm25模型的动态更新
    def _loda_data(self):
        # 加载数据
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        # 从 Redis 获取原始问题
        self.original_questions = self.redis_client.get_data(original_key)
        # 从 Redis 获取分词问题
        tokenized_questions = self.redis_client.get_data(tokenized_key)
        # 如果 Redis 中没有数据，从 MySQL 加载
        if not self.original_questions or not tokenized_questions:
            # 从 MySQL 获取问题
            self.original_questions = self.mysql_client.fetch_questions()
            if not self.original_questions:
                # 记录无问题警告
                self.logger.warning("未加载到问题")
                return
            # 分词问题
            tokenized_questions = [preprocess_text(q[0]) for q in self.original_questions]
            # 存储原始问题到 Redis
            self.redis_client.set_data(original_key, [(q[0]) for q in self.original_questions])
            # 存储分词问题到 Redis
            self.redis_client.set_data(tokenized_key, tokenized_questions)
        # 设置问题列表
        self.questions = tokenized_questions
        # 初始化 BM25 模型
        self.bm25 = BM25Okapi(self.questions)
        # 记录 BM25 初始化成功
        self.logger.info("BM25 模型初始化完成")

    #归一化方法，将BM25的打分表达为概率形式，用于搜索方法的阈值判断
    def _softmax(self,score):
        #对分数进行归一化，总和为一的概率分布表达形式
        exp_score = np.exp(score - np.max(score))
        return exp_score / np.sum(exp_score)

    #搜索方法，用于处理查询、计算相似度、并返回匹配到的文本。
    def search(self,query,threshold=0.85):
        """

        :param query:查询用文本
        :param threshold:匹配相似度的阈值
        :return:匹配成功时：(答案数据,False),匹配失败时(None,True)   布尔类型表示是否为不存在于redis与mysql的查询请求，将进入RAG流程
        """

        if not query or not isinstance(query,str):
            self.logger.error("Query is not a string")
            return None ,True

        #todo 优化用户提问到redis查询的方式，完全使用字符串匹配效率太低
        cache_answer = self.redis_client.get_answer(query)
        print(cache_answer)
        if cache_answer:
            return cache_answer ,False
        #缓存未命中,走数据库查询
        try:
            query_tokens = preprocess_text(query)
            #计算与数据库中所有问题的BM25相似度并进行归一化处理
            score = self.bm25.get_scores(query_tokens)
            softmax_score = self._softmax(score)
            #找到最高分和对应的index
            top_idx= softmax_score.argmax()
            top_score = softmax_score[top_idx]
            logger.info(f"问题内容:"+query+f",匹配得到的最高概率为{top_score}")
            if top_score > threshold:
                original_question = self.original_questions(top_idx)
                answer = self.redis_client.fetch_answer(original_question)
                if answer:
                    logger.info("大于相似度阈值的mysql命中，作为高可靠问题存入redis")
                    self.redis_client.set_data(f'answer:{query}', answer)
                    return answer , False
            self.logger.info(f'无可靠答案，数据库检索失败，最高匹配度{top_score}')
            return None , True


        except Exception as e:
            self.logger.error(f'数据库查询错误：{e}')
        return None ,True






if __name__ == "__main__":
    redis_client = RedisClient()
    mysql_client = MySQLClient()
    bm25 = BM25Search(mysql_client,redis_client)
    bm25.search("喜喜啊",threshold=0.85)