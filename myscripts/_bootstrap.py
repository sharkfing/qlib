"""统一配置 myscripts 脚本的项目模块搜索路径。"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ═══ 允许用户脚本直接导入项目根目录下的 examples、qlib 等模块 ═══
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
