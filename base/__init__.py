#便于其它目录导入自定义模块
import os
import sys

module_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(module_dir)

if module_dir not in sys.path:
    sys.path.append(0,module_dir)

if project_root not in sys.path:
    sys.path.append(0,project_root)

from config import config
from logger import setup_logging