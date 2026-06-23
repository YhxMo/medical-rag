"""RAG医学影像问答系统配置"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "pdfs"
EXCLUDED_PDF_DIR = ROOT / "data" / "excluded_pdfs"
VECTOR_DB_DIR = ROOT / "vector_db"
OUTPUTS_DIR = ROOT / "outputs"

# 目录自动创建
for d in [DATA_DIR, EXCLUDED_PDF_DIR, VECTOR_DB_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 已知存在文本抽取/单位风险的资料，暂不纳入正式知识库
EXCLUDED_PDF_NAMES = {
    "Chinese_Pulmonary_Nodule_Consensus_2018_Interpretation.pdf",
}


def load_local_env(env_path: str | Path | None = None) -> None:
    """Load local .env values without overriding real environment variables."""
    env_file = Path(env_path) if env_path is not None else ROOT / ".env"
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if not key:
            continue

        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_local_env()

# DeepSeek API 配置 (OpenAI 兼容接口)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# HuggingFace 镜像配置（国内用户需设置 HF_ENDPOINT 环境变量使用镜像）
# PowerShell: $env:HF_ENDPOINT = "https://hf-mirror.com"
# 或自动检测：如果未设置环境变量且无法连接 hf.co，则尝试使用镜像
_HF_ENDPOINT = os.environ.get("HF_ENDPOINT")
if _HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = _HF_ENDPOINT

# 嵌入模型配置 (本地运行)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DEVICE = "cpu"  # 或 "cuda"
EMBEDDING_LOCAL_FILES_ONLY = os.environ.get(
    "EMBEDDING_LOCAL_FILES_ONLY",
    "true",
).lower() not in {"0", "false", "no"}

# 文档分块配置
CHUNK_SIZE = 500       # 每个chunk的字符数
CHUNK_OVERLAP = 50     # chunk重叠字符数

# 检索配置
RETRIEVAL_TOP_K = 4    # 检索返回的文档片段数

# 评估配置
EVAL_QUESTIONS_FILE = OUTPUTS_DIR / "eval_questions.json"
EVAL_RESULTS_FILE = OUTPUTS_DIR / "eval_results.json"
