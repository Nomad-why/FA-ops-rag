# 用于BM25检索的jieba切词工具
import jieba
from base import logger

def preprocess_text(text):
    # 预处理文本
    logger.info("进行文本的切词预处理")
    try:
        # 分词并转换为小写
        return jieba.lcut(text.lower())
    except AttributeError as e:
        # 记录预处理失败
        logger.error(f"文本预处理失败: {e}")
        # 返回空列表
        return []