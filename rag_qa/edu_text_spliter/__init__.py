from .edu_chinese_recursive_text_splitter import *

# 条件导入需要额外依赖的模块
try:
    from .edu_model_text_spliter import *
except ImportError:
    pass