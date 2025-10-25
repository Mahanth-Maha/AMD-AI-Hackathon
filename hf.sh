#!/bin/bash
# https://chatgpt.com/share/687b497d-908c-8008-b0fd-3b67d22800b2
set -e

# ====== USER CONFIGURATION ======
HF_USERNAME="myalla"                    # Replace this with your Hugging Face username
REPO_NAME="qwen_aaipl_IISc_Agents"                          # Ensure your LoRA parameters are merged here. E.g., qwen_aaipl_team1
MODEL_DIR="/jupyter-tutorial/AAIPL_129_212_191_46/ckpt/ckpt_q_agent"  # Folder where checkpoint files are stored
PRIVATE=false                                     # true or false
COMMIT_MESSAGE="Initial model upload"
# ==================================

# Step 1: Install dependencies
echo "Installing dependencies..."
apt-get update && apt-get install -y git-lfs && git lfs install

# Step 2: Login to Hugging Face (TOKEN required once)
echo "Logging into Hugging Face CLI..."
huggingface-cli whoami &> /dev/null || huggingface-cli login

# Step 3: Create Hugging Face repo (ONLY if it doesn't exist)
echo "Creating repo on Hugging Face Hub..."
# You can copy the pythonic part and run it in a notebook cell too.
python3 - <<EOF
from huggingface_hub import HfApi

api = HfApi()
repo_id = "${HF_USERNAME}/${REPO_NAME}"
try:
    api.repo_info(repo_id=repo_id)
    print(f"Repo '{repo_id}' already exists.")
except:
    api.create_repo(name="${REPO_NAME}", private=${PRIVATE})
    print(f"Created new repo: {repo_id}")
EOF

# Step 4: Git-based upload
echo "Preparing Git repo and uploading checkpoint..."

cd "$MODEL_DIR"

git init
git lfs install
git remote add origin "https://huggingface.co/${HF_USERNAME}/${REPO_NAME}"
git pull origin main || echo "Remote branch doesn't exist yet. Skipping pull."
git add .
git commit -m "$COMMIT_MESSAGE"
git push origin main

echo "✅ Upload complete: https://huggingface.co/${HF_USERNAME}/${REPO_NAME}"
