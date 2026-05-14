import os
import sys
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def diagnose():
    # 当前工作目录
    cwd = os.getcwd()
    # 根目录文件列表
    root_files = os.listdir(cwd)
    # scripts 目录是否存在，里面有哪些文件
    scripts_path = os.path.join(cwd, "scripts")
    scripts_exists = os.path.exists(scripts_path)
    scripts_files = os.listdir(scripts_path) if scripts_exists else []
    # 检查 savant_client.py 是否存在
    savant_path = os.path.join(scripts_path, "savant_client.py")
    savant_exists = os.path.exists(savant_path)
    # Python 路径
    python_path = sys.path

    return {
        "cwd": cwd,
        "root_files": root_files,
        "scripts_exists": scripts_exists,
        "scripts_files": scripts_files,
        "savant_client_exists": savant_exists,
        "python_path": python_path
    }
    return summary
