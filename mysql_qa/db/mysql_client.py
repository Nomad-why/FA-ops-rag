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











if __name__ == "__main__":
    mysql_client = MySQLClient()