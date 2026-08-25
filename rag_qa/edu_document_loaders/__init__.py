import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入配置
from base.config import config

# 添加配置中的路径到 sys.path
# sys.path.append(config.EDU_DOCUMENT_LOADERS_DIR)

# 条件导入各个文档加载器，允许在缺少依赖时继续运行
try:
    from .edu_docloader import *
except ImportError:
    pass

try:
    from .edu_pptloader import *
except ImportError:
    pass

try:
    from .edu_imgloader import *
except ImportError:
    pass

try:
    from .edu_pdfloader import *
except ImportError:
    pass
