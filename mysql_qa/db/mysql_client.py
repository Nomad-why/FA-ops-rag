#用于MYSQL的基础操作

import os , sys
import pymysql
import pandas as pd

#获取当前文件所在目录
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0,project_root)

#导入配置与日志模块
from base import config , logger


class MySQLClient:
    def __init__(self):
        self.logger = logger
        try:
            # crate MYSQL connection
            self.connection = pymysql.connect(
                host = config.MYSQL_HOST,
                user = config.MYSQL_USER,
                password = config.MYSQL_PASSWORD,
                database = config.MYSQL_DATABASE,
            )

            self.cursor = self.connection.cursor()
            self.logger.info("MYSQL CONNECT SUCCESSFUL")

        except pymysql.MySQLError as e:
            self.logger.error(f'mysql connect error: {e}')

    def create_table(self):
        create_table_query = '''
                             CREATE TABLE IF NOT EXISTS jpkb \
                             ( \
                                 id \
                                 INT \
                                 AUTO_INCREMENT \
                                 PRIMARY \
                                 KEY, \
                                 subject_name \
                                 VARCHAR \
                             ( \
                                 20 \
                             ),
                                 question VARCHAR \
                             ( \
                                 1000 \
                             ),
                                 answer VARCHAR \
                             ( \
                                 1000 \
                             )) \
                             '''
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            self.logger.info("表创建成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"表创建失败: {e}")
            raise

    def insert_data(self, csv_path):
        try:
            data = pd.read_csv(csv_path)
            # 将 DataFrame 转为元组列表
            data_tuples = [tuple(x) for x in data[['学科名称', '问题', '答案']].to_numpy()]
            insert_query = "INSERT INTO jpkb (subject_name, question, answer) VALUES (%s, %s, %s)"

            # 假设每次处理 5000 条数据
            ##todo,加入配置文件便与修改
            chunk_size = 5000
            for i in range(0, len(data_tuples), chunk_size):
                chunk = data_tuples[i: i + chunk_size]
                self.cursor.executemany(insert_query, chunk)

            # 所有分块都执行完后，再统一提交
            self.connection.commit()
            self.logger.info("数据插入成功")
        except Exception as e:
            self.logger.error(f"数据插入失败: {e}")
            self.connection.rollback()
            raise

    def fetch_questions(self):
        ##todo结合后续的实际用途，解决数据量较大时的一次性全部返回写法导致的效率问题
        # 获取所有问题
        try:
            # 执行查询
            self.cursor.execute("SELECT question FROM jpkb")
            results = self.cursor.fetchall()
            self.logger.info("成功获取问题")
            return results
        except pymysql.MySQLError as e:
            self.logger.error(f"查询失败: {e}")
            # 返回空列表
            return []

    def fetch_answer(self, question):
        # 获取指定问题的答案
        try:
            # 执行查询
            self.cursor.execute("SELECT answer FROM jpkb WHERE question=%s", (question,))
            # 获取结果
            result = self.cursor.fetchone()
            # 返回答案或 None
            return result[0] if result else None
        except pymysql.MySQLError as e:
            # 记录答案获取失败
            self.logger.error(f"答案获取失败: {e}")
            # 返回 None
            return None

    def close(self):
        # 关闭数据库连接
        try:
            # 关闭连接
            self.connection.close()
            # 记录关闭成功
            self.logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            # 记录关闭失败
            self.logger.error(f"关闭连接失败: {e}")











if __name__ == "__main__":
    mysql_client = MySQLClient()
    mysql_client.create_table()
    mysql_client.insert_data('../data/test.csv')