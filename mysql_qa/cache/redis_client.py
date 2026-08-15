#Redis缓存：只存储高可靠性的问答，搜索得到匹配相似度大于85%时存入redis。
import redis
import json
from base import config, logger


class RedisClient:
    def __init__(self):
        # 初始化日志
        self.logger = logger
        try:
            # 连接 Redis
            self.client = redis.StrictRedis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                db=config.REDIS_DB,
                decode_responses=True
            )
            self.logger.info("Redis 连接成功")
        except redis.RedisError as e:
            self.logger.error(f"Redis 连接失败: {e}")
            raise

    def set_data(self, key, value):
        # 存储数据到 Redis
        try:
            # 存储 JSON 数据
            self.client.set(key, json.dumps(value))
            # 记录存储成功
            self.logger.info(f"存储数据到 Redis: {key}")
        except redis.RedisError as e:
            # 记录存储失败
            self.logger.error(f"Redis 存储失败: {e}")

    def get_data(self, key):
        # 从 Redis 获取数据
        try:
            data = self.client.get(key)
            # 返回解析后的 JSON 数据或 None
            return json.loads(data) if data else None
        except redis.RedisError as e:
            # 记录获取失败并返回None
            self.logger.error(f"Redis 获取失败: {e}")
            return None

    def get_answer(self, query):
        # 获取查询的缓存答案
        try:
            # 从 Redis 获取答案
            answer = self.client.get(f"answer:{query}")
            if answer:
                # 记录获取成功，返回答案
                self.logger.info(f"从 Redis 获取答案: {query}")
                return answer
            # 返回 None
            return None
        except redis.RedisError as e:
            # 记录查询失败并返回None
            self.logger.error(f"Redis 查询失败: {e}")
            return None


if __name__ == '__main__':
    redcli = RedisClient()