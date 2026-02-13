import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

BASE_DIR = Path("/workspace/experiment/Clarifying-Ambiguity-Project")


def push_data(folder_path: str):

    p = Path(folder_path)
    path_in_repo = '/'.join(p.parts[-1:])

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_folder(
        folder_path=folder_path,
        repo_id="jhpark0012/ClarifyingAmbiguity",
        repo_type="dataset",
        path_in_repo=path_in_repo,
    )

    return "Done : push_data"


def pull_data(folder_path: str):

    p = Path(folder_path.rstrip("/"))
    repo_subdir = "/".join(p.parts[-1:])

    os.makedirs(folder_path, exist_ok=True)

    snapshot_download(
        repo_id="jhpark0012/ClarifyingAmbiguity",
        repo_type="dataset",
        allow_patterns=[f"{repo_subdir}/**"],
        local_dir=folder_path,
        token=os.environ["HF_TOKEN"],
    )

    return "Done : pull_data"
