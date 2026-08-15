#配置文件管理 -- 配置文件路径为./base/config.ini
import configparser
import os

#当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
#当前文件所在目录绝对路径
current_dir_path = os.path.dirname(current_file_path)
#项目根目录绝对路径
project_root = os.path.dirname(current_dir_path)
#拼接配置文件的完整路径
config_file_path = os.path.join(project_root, 'config.ini')


class Config :
    def __init__(self,config_file=config_file_path):
        #创建配置文件实例
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        #解析MySQL配置
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='1wuqingtieshou')
        self.MYSQL_DATABASE = self.config.get('mysql', 'datebase', fallback='qa_system')

        #解析Redis数据库配置
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        self.REDIS_PORT = self.config.get('redis', 'port', fallback=6379)
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='None')

        #解析日志配置
        self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log')

config = Config()


if __name__ == '__main__':
    print(config.LOG_FILE)
