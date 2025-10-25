# Final LogicTrainer with Multi-Topic and Custom Dataset Support

import os
import torch
import re
import json
import time
from typing import List, Dict, Optional
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
from difflib import SequenceMatcher

class LogicTrainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        self._inference_model = None
        self._inference_tokenizer = None

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

        self._setup_environment()
        self._display_config()

    def _get_prompt_for_topic(self, topic):
        base = self.topic_to_prompt.get(topic, "You are an expert logical reasoner.")
        return (
            f"{base} Your task is to answer multiple-choice logic questions."
            f" Provide your reasoning between {self.reasoning_start} and {self.reasoning_end},"
            f" and your final answer between {self.solution_start} and {self.solution_end}."
            f" Only one letter (A/B/C/D) is allowed inside the answer tag."
        )

    def _setup_environment(self):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", self.args.gpu_ids)
        if self.args.training_type == 'grpo':
            os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "1")
            os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
        print(f"Using devices: {torch.cuda.device_count()} GPUs")

    def _display_config(self):
        print("=" * 60)
        print(f"Training Type: {self.args.training_type}")
        print(f"Model: {self.args.model_name}")
        print(f"Epochs: {self.args.num_train_epochs}, LR: {self.args.learning_rate}, Batch: {self.args.per_device_train_batch_size}")
        print("=" * 60)

    def load_dataset(self):
        with open(self.args.dataset_file, 'r') as f:
            data = json.load(f)
        raw = data if isinstance(data, list) else [data]

        if self.args.training_type == 'sft':
            tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            formatted = []
            for item in raw:
                topic = item.get("topic", "Puzzles involving generations and family tree logic")
                question = item.get("question", "")
                choices = item.get("choices", [])
                full_question = f"{question}\n" + "\n".join(choices)
                answer = item.get('answer', '')
                reasoning = item.get('explanation', 'Let us break this down step by step.')
                system_prompt = self._get_prompt_for_topic(topic)
                model_output = f"{self.reasoning_start}{reasoning}{self.reasoning_end}{self.solution_start}{answer}{self.solution_end}"
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': full_question},
                    {'role': 'assistant', 'content': model_output}
                ]
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                formatted.append({"text": formatted_text})
            self.dataset = Dataset.from_list(formatted)
        else:
            formatted = []
            for item in raw:
                topic = item.get("topic", "Puzzles involving generations and family tree logic")
                question = item.get("question", "")
                choices = item.get("choices", [])
                full_question = f"{question}\n" + "\n".join(choices)
                answer = item.get('answer', '')
                system_prompt = self._get_prompt_for_topic(topic)
                formatted.append({
                    'prompt': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': full_question}
                    ],
                    'answer': answer
                })
            self.dataset = Dataset.from_list(formatted)
