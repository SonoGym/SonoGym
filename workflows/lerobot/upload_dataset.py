from huggingface_hub import upload_folder
from huggingface_hub import HfApi

api = HfApi()

api.create_branch(
    repo_id="yunkao/us-guidance",  # or model name
    repo_type="dataset",  # or "model"
    branch="v2",
    exist_ok=True  # Optional: don’t throw error if it already exists
)

upload_folder(
    folder_path="/home/yunkao/git/IsaacLabExtensionTemplate/lerobot-dataset/Isaac-robot-US-guidance-v0",
    path_in_repo=".",  # Folder inside the repo
    repo_id="yunkao/us-guidance",
    repo_type="dataset",
    revision="v2",
)