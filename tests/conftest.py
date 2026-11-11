import os
from dotenv import load_dotenv

# 这个 hook 会在所有测试开始前运行
def pytest_configure(config):
    # 1. 先加载基础环境配置 (local)
    base_env = os.path.join("docker", "local", ".env_local")
    load_dotenv(base_env)
    
    # 2. 再加载测试环境配置 (覆盖 local 中的相同变量)
    test_env = os.path.join("tests", ".env_test")
    load_dotenv(test_env, override=True) # override=True 是实现“覆盖”的关键
    print("`.env_test` loaded")
    