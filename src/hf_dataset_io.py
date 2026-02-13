import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

BASE_DIR = Path("/workspace/experiment/Clarifying-Ambiguity-Project")
load_dotenv(BASE_DIR / ".env")   # HF_TOKEN 로드


def push_data(folder_path: str):

    p = Path(folder_path)
    path_in_repo = '/'.join(p.parts[-2:])

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_folder(
        folder_path=folder_path,
        repo_id="QILAB-yj/yj-data",
        repo_type="dataset",
        path_in_repo=path_in_repo,
    )

    return "Done : push_data"


def pull_data(folder_path: str):

    p = Path(folder_path.rstrip("/"))
    repo_subdir = "/".join(p.parts[-2:])  # capsule/001

    os.makedirs(folder_path, exist_ok=True)

    snapshot_download(
        repo_id="QILAB-yj/yj-data",
        repo_type="dataset",
        allow_patterns=[f"{repo_subdir}/**"],
        local_dir=folder_path,
        token=os.environ["HF_TOKEN"],
    )

    return "Done : pull_data"
