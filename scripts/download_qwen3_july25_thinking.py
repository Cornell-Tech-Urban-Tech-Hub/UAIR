import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "Qwen/Qwen3-30B-A3B-Thinking-2507",
    local_dir = "/share/ju/matt/sensing-ai-risks/models/Qwen3-30B-A3B-Thinking-2507"
)