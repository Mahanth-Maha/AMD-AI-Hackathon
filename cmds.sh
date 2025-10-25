
# Clear any cached memory
python -c "import torch; torch.cuda.empty_cache()"

nohup python -m train_q_agent \
    --mode train \
    --model_name /jupyter-tutorial/hf_models/Qwen3-4B \
    --dataset_file /jupyter-tutorial/AAIPL_maha/data/test-chota.json \
    --output_dir q_agent_sft_v3 \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --disable_wandb > q_agent_sft_v3.log 2>&1 &

tail -f q_agent_sft_v3.log




nohup python -m train_a_agent \
    --mode train \
    --model_name /jupyter-tutorial/hf_models/Qwen3-4B \
    --dataset_file /jupyter-tutorial/AAIPL_maha/data/test-chota.json \
    --output_dir a_agent_grpo_v3 \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --disable_wandb > a_agent_grpo_v3.log 2>&1 &

tail -f a_agent_grpo_v3.log


python -m trainer_v3 \
    --training_type grpo \
    --model_name "/jupyter-tutorial/hf_models/Qwen3-4B" \
    --dataset_file "/jupyter-tutorial/AAIPL_maha/data/train.json" \
    --output_dir "a_agent_grpo" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_seq_length 1024 \
    --max_prompt_length 512 \
    --disable_wandb > a_agent_grpo_v2.log 2>&1 &

tail -f a_agent_grpo_v2.log







# Q Agent SFT 
nohup python -m train_q_agent \
    --mode train \
    --model_name /jupyter-tutorial/hf_models/Qwen3-4B \
    --dataset_file /jupyter-tutorial/AAIPL_maha/data/train.json \
    --output_dir ckpt_qAgent \
    --learning_rate 2e-5 \
    --num_train_epochs 30 \
    --per_device_train_batch_size 4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --disable_wandb > log_qAgent.log 2>&1 &

tail -f log_qAgent.log


# A Agent GRPO
python -m trainer_v3 \
    --training_type grpo \
    --model_name "/jupyter-tutorial/hf_models/Qwen3-4B" \
    --dataset_file "/jupyter-tutorial/AAIPL_maha/data/train.json" \
    --output_dir "ckpt_aAgent" \
    --num_train_epochs 30 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_seq_length 1024 \
    --max_prompt_length 512 \
    --disable_wandb > log_aAgent.log 2>&1 &

tail -f log_aAgent.log