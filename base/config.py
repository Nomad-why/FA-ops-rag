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
        self.config.read(config_file , encoding='utf-8')

        #RAG工程中模型、数据、文档加载器路径配置
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/models')
        self.EDU_DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/edu_document_loaders')

        #解析MySQL配置
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='1wuqingtieshou')
        self.MYSQL_DATABASE = self.config.get('mysql', 'datebase', fallback='qa_system')

        #解析Redis数据库配置
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        self.REDIS_PORT = self.config.get('redis', 'port', fallback=6379)
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='None')
        self.REDIS_DB = self.config.get('redis', 'database', fallback='0')

        # Milvus 配置
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME',self.config.get('milvus', 'database_name'))
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME',self.config.get('milvus', 'collection_name'))

        # LLM 配置
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen-plus')
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', self.config.get('llm', 'dashscope_api_key'))
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url',fallback='https://dashscope.aliyuncs.com/compatible-mode/v1')

        # 检索参数
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        self.VALID_SOURCES = eval(self.config.get('app', 'valid_sources'))
        #解析日志配置
        self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log')

config = Config()


if __name__ == '__main__':
    print(config.LOG_FILE)
