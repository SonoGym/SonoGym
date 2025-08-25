from huggingface_hub import HfApi

hub_api = HfApi()
hub_api.create_tag("yunkao/SonoGym_lerobot_dataset", tag="v1", repo_type="dataset")