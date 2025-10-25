import os
import torch
import argparse
import re
import json
import time
from typing import Optional, List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback
)
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
from peft import LoraConfig, PeftModel
import wandb

class AgentTrainer:
    """
    Unified trainer for fine-tuning agents for the AAIPL competition.
    Supports:
    - SFT for Q-Agent (Question Generation)
    - GRPO for A-Agent (Question Answering)
    """
    
    def __init__(self, args):
        """Initialize the trainer with configuration arguments."""
        self.args = args
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        
        # Inference model cache
        self._inference_model = None
        self._inference_tokenizer = None

        # A-Agent specific constants
        self.reasoning_start = "<reasoning>"
        self.reasoning_end = "</reasoning>"
        self.solution_start = "<answer>"
        self.solution_end = "</answer>"

        # Updated topics with exact values used in your dataset
        self.topic_to_prompt = {
            "Puzzles involving generations and family tree logic": "You are an expert in solving complex family and generation-based logic puzzles.",
            "Truth-teller and Liar Problems": "You are an expert in solving puzzles involving truth-tellers and liars.",
            "Seating Arrangements (Linear, Circular)": "You are an expert in seating arrangement problems, including linear and circular arrangements."
        }
        
        # System prompts tailored to each agent type
        self.a_agent_system_prompt = f"""
You are an expert in logical reasoning and complex problem-solving.
Your task is to answer multiple-choice questions (MCQs) on topics like 'Puzzles involving generations and family tree logic', 'Truth-teller and Liar Problems', and 'Seating Arrangements (Linear, Circular)'.
First, think about the problem and provide your step-by-step reasoning. Place it between {self.reasoning_start} and {self.reasoning_end}.
Then, provide your final answer between {self.solution_start} and {self.solution_end}.
The final answer must be a single capital letter (A, B, C, or D) corresponding to the correct choice.
Follow the JSON format strictly. Do not include any additional text outside of these tags.
"""
        self.q_agent_system_prompt = """
You are a creative and expert puzzle maker.
Your task is to generate a high-quality, multiple-choice question (MCQ) in a valid JSON format.
The user will provide a topic, and you must create a question with four choices (A, B, C, D), a single-letter correct answer, and a brief explanation.
The JSON must contain the keys: "topic", "question", "choices", "answer", and "explanation".
The "choices" field must be an array of four strings, each starting with its corresponding letter (e.g., "A) ...").
The "answer" must be a single capital letter.
"""
        
        self._setup_environment()
        
        # Display configuration
        self._display_config()
    
    def _setup_environment(self):
        """Setup environment variables and GPU configuration."""
        # GPU selection - parse gpu_ids from args
        gpu_ids = [int(x.strip()) for x in self.args.gpu_ids.split(',') if x.strip().isdigit()]
        if not gpu_ids:
            gpu_ids = [0]  # Default to GPU 0 if no valid IDs provided
        
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ','.join(map(str, gpu_ids)))
        
        # Add environment variables for distributed training
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        
        # ROCm optimization flags for vLLM (if using GRPO)
        if self.args.training_type == 'grpo':
            os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "0")
            os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
        
        print(f"PyTorch detected number of available devices: {torch.cuda.device_count()}")
    
    def _display_config(self):
        """Display training configuration."""
        print("=" * 60)
        print(f"AMD AI PREMIER LEAGUE - {self.args.agent_type.upper()} TRAINER")
        print("=" * 60)
        print(f"Agent Type: {self.args.agent_type}")
        print(f"Training Type: {self.args.training_type}")
        print(f"Mode: {self.args.mode}")
        print(f"Model: {self.args.model_name}")
        print(f"Output directory: {self.args.output_dir}")
        print(f"Dataset file: {self.args.dataset_file}")
        print(f"Learning rate: {self.args.learning_rate}")
        print(f"Epochs: {self.args.num_train_epochs}")
        print(f"Batch size: {self.args.per_device_train_batch_size}")
        print(f"LoRA rank: {self.args.lora_r}")
        print(f"LoRA alpha: {self.args.lora_alpha}")
        print(f"Max sequence length: {self.args.max_seq_length}")
        print(f"GPU IDs: {self.args.gpu_ids}")
        print("=" * 60)

    def load_dataset(self):
        """Load and process the dataset based on the agent type."""
        print(f"Loading dataset from: {self.args.dataset_file}")
        try:
            with open(self.args.dataset_file, 'r', encoding='utf-8') as f:
                raw_items = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading dataset: {e}")
            raise
            
        print(f"Loaded {len(raw_items)} raw items.")

        if self.args.agent_type == 'q_agent':
            self.dataset = self._format_q_agent_sft_dataset(raw_items)
        elif self.args.agent_type == 'a_agent':
            if self.args.training_type == 'sft':
                self.dataset = self._format_a_agent_sft_dataset(raw_items)
            else: # grpo
                self.dataset = self._format_a_agent_grpo_dataset(raw_items)
        else:
            raise ValueError("Invalid agent_type specified.")

    def _format_q_agent_sft_dataset(self, raw_items: List[Dict]) -> Dataset:
        """Format data for Q-Agent SFT. Input: topic, Output: full JSON."""
        tokenizer = self._get_tokenizer()
        formatted_items = []
        for item in raw_items:
            # The user prompt is just the topic
            user_prompt = f"Generate a question for the topic: {item.get('topic', 'Logical Reasoning')}"
            
            # The assistant's response is the well-formed JSON string
            assistant_response = json.dumps({
                "topic": item.get("topic"),
                "question": item.get("question"),
                "choices": item.get("choices"),
                "answer": item.get("answer"),
                "explanation": item.get("explanation")
            }, indent=4)

            chat_messages = [
                {'role': 'system', 'content': self.q_agent_system_prompt},
                {'role': 'user', 'content': user_prompt},
                {'role': 'assistant', 'content': assistant_response}
            ]
            
            formatted_text = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=False
            )
            formatted_items.append({"text": formatted_text})

        print(f"Created {len(formatted_items)} Q-Agent SFT samples.")
        return Dataset.from_list(formatted_items)

    def _format_a_agent_sft_dataset(self, raw_items: List[Dict]) -> Dataset:
        """Format data for A-Agent SFT. Input: question, Output: reasoning/answer."""
        tokenizer = self._get_tokenizer()
        formatted_items = []
        for item in raw_items:
            question_text = item['question'] + "\n" + "\n".join(item.get('choices', []))
            
            # Model's expected output
            model_completion = (
                f"{self.reasoning_start}{item.get('explanation', 'Let me think step-by-step.')}{self.reasoning_end}"
                f"{self.solution_start}{item.get('answer', '')}{self.solution_end}"
            )

            chat_messages = [
                {'role': 'system', 'content': self.a_agent_system_prompt},
                {'role': 'user', 'content': question_text},
                {'role': 'assistant', 'content': model_completion}
            ]
            
            formatted_text = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=False
            )
            formatted_items.append({"text": formatted_text})
            
        print(f"Created {len(formatted_items)} A-Agent SFT samples.")
        return Dataset.from_list(formatted_items)

    def _format_a_agent_grpo_dataset(self, raw_items: List[Dict]) -> Dataset:
        """Format data for A-Agent GRPO training."""
        formatted_items = []
        for item in raw_items:
            question_text = item['question'] + "\n" + "\n".join(item.get('choices', []))
            
            formatted_items.append({
                'prompt': [
                    {'role': 'system', 'content': self.a_agent_system_prompt},
                    {'role': 'user', 'content': question_text}
                ],
                'labels': item.get('answer', '')
            })
            
        print(f"Created {len(formatted_items)} A-Agent GRPO samples.")
        return Dataset.from_list(formatted_items)

    def _get_tokenizer(self):
        """Helper to get a tokenizer instance."""
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer for training."""
        print(f"Loading base model: {self.args.model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.config.use_cache = False
        
        self.tokenizer = self._get_tokenizer()
        self.tokenizer.model_max_length = self.args.max_seq_length
        print("Model and tokenizer loaded successfully.")

    def setup_peft_config(self) -> LoraConfig:
        """Setup LoRA configuration."""
        return LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )

    def train(self):
        """Main training function."""
        self.load_dataset()
        self.setup_model_and_tokenizer()
        
        if self.args.training_type == 'sft':
            self.train_sft()
        elif self.args.training_type == 'grpo':
            self.train_grpo()
        else:
            raise ValueError(f"Unsupported training_type: {self.args.training_type}")

    def train_sft(self):
        """Train using Supervised Fine-Tuning."""
        print(f"Starting SFT training for {self.args.agent_type.upper()}...")
        
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            bf16=True,
            save_strategy="epoch",
            logging_steps=10,
            lr_scheduler_type="cosine",
            report_to="wandb" if not self.args.disable_wandb else "none",
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=self.dataset,
            peft_config=self.setup_peft_config(),
            dataset_text_field="text",
            max_seq_length=self.args.max_seq_length,
        )
        
        self.trainer.train()
        self.trainer.save_model(self.args.output_dir)
        print("SFT training completed and model saved.")

    # In the train_grpo method

    def train_grpo(self):
        """Train using Group Relative Policy Optimization (for A-Agent)."""
        if self.args.agent_type != 'a_agent':
            raise ValueError("GRPO is only configured for the A-Agent.")
            
        print("Starting GRPO training for A_AGENT...")
        
        config = GRPOConfig(
            output_dir=self.args.output_dir,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            bf16=True,
            logging_steps=1,
            save_strategy="epoch",
            report_to="wandb" if not self.args.disable_wandb else "none",
            # GRPO specific
            beta=0.1,
            max_prompt_length=self.args.max_prompt_length,
            max_completion_length=self.args.max_seq_length - self.args.max_prompt_length,
            use_vllm=True,
            vllm_gpu_memory_utilization=0.9,
            # Explicitly name the label column
            label_names=["labels"],
            remove_unused_columns=False # Important: keeps 'labels' for the reward function
        )
        
        # Use reward_funcs (as a list) and processing_class
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=config,
            train_dataset=self.dataset,
            peft_config=self.setup_peft_config(),
            reward_funcs=[self._grpo_reward_func],
        )
        
        self.trainer.train()
        self.trainer.save_model(self.args.output_dir)
        print("GRPO training completed and model saved.")

    def extract_xml_answer(self, text: str) -> str:
        """Extracts the single-letter answer from the XML-like response."""
        match = re.search(r"<answer>([A-D])</answer>", text)
        return match.group(1) if match else ""

    # def _grpo_reward_func(self, completions: List[str], **kwargs) -> torch.FloatTensor:
    #     """Combined reward function for GRPO training on the A-Agent."""
    #     rewards = []
    #     # 'answer' is a key passed by the GRPOTrainer containing the ground truth
    #     ground_truth_answers = kwargs["answer"]

    #     for completion, truth in zip(completions, ground_truth_answers):
    #         total_reward = 0.0
            
    #         # Reward 1: Format Correctness
    #         # Check if both reasoning and answer tags are present.
    #         has_reasoning_tags = self.reasoning_start in completion and self.reasoning_end in completion
    #         has_answer_tags = self.solution_start in completion and self.solution_end in completion
    #         if has_reasoning_tags and has_answer_tags:
    #             total_reward += 1.0  # Base reward for correct format
            
    #         # Reward 2: Answer Correctness
    #         extracted_answer = self.extract_xml_answer(completion)
    #         if extracted_answer and extracted_answer == truth:
    #             total_reward += 2.0  # Strong reward for correct answer

    #         # Penalty: Excessive Length
    #         if len(completion) > 1024:
    #             total_reward -= 0.5

    #         rewards.append(total_reward)
            
    #     return torch.tensor(rewards, dtype=torch.float32)
    # In the AgentTrainer class

    def _grpo_reward_func(self, completions: List[str], **kwargs) -> torch.FloatTensor:
        """Combined reward function for GRPO training on the A-Agent."""
        rewards = []
        # Expect the ground truth answers in the 'labels' key now
        ground_truth_answers = kwargs["labels"]

        for completion, truth in zip(completions, ground_truth_answers):
            total_reward = 0.0
            
            # Reward 1: Format Correctness
            has_reasoning_tags = self.reasoning_start in completion and self.reasoning_end in completion
            has_answer_tags = self.solution_start in completion and self.solution_end in completion
            if has_reasoning_tags and has_answer_tags:
                total_reward += 1.0

            # Reward 2: Answer Correctness
            extracted_answer = self.extract_xml_answer(completion)
            if extracted_answer and extracted_answer == truth:
                total_reward += 2.0

            # Penalty: Excessive Length
            if len(completion) > 1024:
                total_reward -= 0.5

            rewards.append(total_reward)
            
        return torch.tensor(rewards, dtype=torch.float32)

def parse_args():
    parser = argparse.ArgumentParser(description="AAIPL Agent Trainer")
    
    # Core arguments
    parser.add_argument("--agent_type", type=str, required=True, choices=["q_agent", "a_agent"], help="Type of agent to train.")
    parser.add_argument("--training_type", type=str, required=True, choices=["sft", "grpo"], help="Training methodology.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"], help="Mode of operation.")
    parser.add_argument("--model_name", type=str, default="/jupyter-tutorial/hf_models/Qwen3-4B", help="Base model name or path.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the training JSON dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints.")
    
    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum prompt length for GRPO.")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=float, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    
    # System arguments
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs to use (e.g., '0,1').")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="aaipl-fine-tuning", help="WandB project name.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate arguments
    if args.agent_type == 'q_agent' and args.training_type != 'sft':
        raise ValueError("Q-Agent training only supports SFT. Please set --training_type sft.")
    if args.training_type == 'grpo' and args.agent_type != 'a_agent':
         raise ValueError("GRPO training is only implemented for the A-Agent.")

    trainer = AgentTrainer(args)
    
    if args.mode == 'train':
        trainer.train()
    else:
        # Inference logic can be added here based on the original script's methods
        print("Inference mode not fully implemented in this version. Please adapt from the original script.")

if __name__ == "__main__":
    main()