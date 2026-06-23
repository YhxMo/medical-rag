"""测试本地配置加载。"""
from __future__ import annotations

import os

from config import load_local_env


def test_load_local_env_reads_dotenv_without_overriding_environment(
    tmp_path,
    monkeypatch,
):
    """本地.env可提供密钥，但不能覆盖已设置的环境变量。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join([
            "# local secrets",
            "DEEPSEEK_API_KEY=local-test-key",
            "DEEPSEEK_MODEL=local-test-model",
        ]),
        encoding="utf-8",
    )

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_MODEL", "existing-model")

    load_local_env(env_file)

    assert os.environ["DEEPSEEK_API_KEY"] == "local-test-key"
    assert os.environ["DEEPSEEK_MODEL"] == "existing-model"
