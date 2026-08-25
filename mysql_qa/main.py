#MYSQL模块的入门程序
from idlelib import query

from base import logger
from retrieval.bm25_search import BM25Search
from cache.redis_client import RedisClient
from db.mysql_client import  MySQLClient
import time

class MysqlQASystem:
    def __init__(self):
        self.logger = logger
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25 = BM25Search(self.mysql_client, self.redis_client)


    def query(self, query):
        """
        mysql模块的查询处理，用文档数据建立的的BM25模型从mysql获取答案，返回结果并记录日志
        :param query: 用户输入的查询文本
        :return:返回找到的答案，或默认提示代表未找到
        """
        start_time = time.time()
        self.logger.info(f"开始处理问答，输入内容为：{query}")
        answer , _ =self.bm25.search(query,threshold=0.85)
        if answer:
            self.logger.info(f"MYSQL命中：{answer}")
            return answer
        else:
            self.logger.info("进入RAG流程")
            return None


def main():
        mysql_qa = MysqlQASystem()
        try:
            print("\n欢迎使用现网问题处理助手")
            print('回复基于现有案例整理+LLM处理得出,内容不保证完全正确,建议参考回复中给出的案例链接详细了解情况,处理问题时注意案例中标注的系统版本适用范围，如发现案例有误可提资料单进行修改。')
            print("目前无记录历史对话功能，需要留存的信息建议复制保存，关闭窗口后可能无法复现")
            print('输入查询内容进行交互，输入exit退出')
            while True:
                query = input("\输入问题内容:").strip()
                if query.lower() == 'exit':
                    logger.info("退出系统")
                    print("over")
                    break
                answer = mysql_qa.query(query)
                print(f'{answer}')




        except Exception as e:
            logger.error(f"系统启动异常：{e}")
        finally:
            mysql_qa.mysql_client.close()


if __name__ == "__main__":
     main()
