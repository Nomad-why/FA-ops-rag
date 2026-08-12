#日志设置
import logging
import os
from base.config import config


def setup_logging(log_file=config.LOG_FILE):
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # 日志器级别设置为info
    logger = logging.getLogger("FA-qa")
    logger.setLevel(logging.INFO)
    # 避免重复添加处理器前提下，分别添加控制台与文件处理器，并设置日志格式
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s %(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # 添加两个处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# 初始化日志器
logger = setup_logging()

if __name__ == '__main__':
    logger.info('greedisgood 500')