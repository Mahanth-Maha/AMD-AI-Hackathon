import os
import argparse
import json
import time
import re
from typing import List, Dict, Optional
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, PeftModel
import torch
import wandb

# import time
# time.sleep(60* 60 * 2) # Wait for 2 hours
# # clear memory
# torch.cuda.empty_cache()

class AnswerAgentTrainer:
    """
    Trainer class for fine-tuning a model to ANSWER aptitude questions using GRPO.
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
        
        # --- XML Tags for Answer Generation ---
        self.reasoning_start = "<reasoning>"
        self.reasoning_end = "</reasoning>"
        self.solution_start = "<answer>"
        self.solution_end = "</answer>"

        self.system_prompt_answering = f"""
        You are an expert in logical reasoning and complex problem-solving.
        Your task is to answer multiple-choice questions (MCQs).
        First, think step-by-step about the problem and provide your detailed working out.
        Place your reasoning between {self.reasoning_start} and {self.reasoning_end} tags.
        Then, provide your final answer, which must be a single capital letter (A, B, C, or D).
        Place the final answer between {self.solution_start} and {self.solution_end} tags.
        Do not include any additional text outside of these tags.
        """
        self._setup_environment()
        self._display_config()

    def _setup_environment(self):
        """Setup environment variables and GPU configuration for GRPO."""
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", self.args.gpu_ids)
        # Add vLLM specific flags for GRPO efficiency
        os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "0")
        os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
        print(f"PyTorch detected number of available devices: {torch.cuda.device_count()}")

    def _display_config(self):
        """Display training configuration."""
        print("=" * 60)
        print("A-AGENT (ANSWER GENERATOR) GRPO TRAINER")
        print("=" * 60)
        # print(f"Training Type: {self.args.training_type}")
        # print(f"Mode: {self.args.mode}")
        # print(f"Model: {self.args.model_name}")
        # print(f"Output directory: {self.args.output_dir}")
        # print(f"Dataset file: {self.args.dataset_file}")
        # print(f"Learning rate: {self.args.learning_rate}")
        # print(f"Epochs: {self.args.num_train_epochs}")
        # print(f"Batch size: {self.args.per_device_train_batch_size}")
        # print(f"LoRA rank: {self.args.lora_r}")
        # print(f"LoRA alpha: {self.args.lora_alpha}")
        # print(f"Max sequence length: {self.args.max_seq_length}")
        # print(f"GPU IDs: {self.args.gpu_ids}")
        # print("=" * 60)

    def _load_raw_dataset(self) -> List[Dict[str, str]]:
        """Load raw dataset items from JSON file."""
        items = []
        try:
            with open(self.args.dataset_file, 'r', encoding='utf-8') as f:
                question_objects = json.load(f)

            for idx, q_obj in enumerate(question_objects, 1):
                question = q_obj.get("question")
                choices = q_obj.get("choices", [])
                answer = q_obj.get("answer")
                
                if question and answer and len(choices) > 0:
                    full_question = f"Question: {question}\n\n" + "\n".join(choices)
                    items.append({
                        'question': full_question,
                        'answer': answer.strip().upper(),
                    })
                else:
                    print(f"Warning: Skipping incomplete question data at index {idx}")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        
        print(f"Successfully loaded {len(items)} questions for training.")
        return items

    def _format_grpo_dataset(self, raw_items: List[Dict[str, str]]) -> Dataset:
        """Format raw items for GRPO training."""
        formatted_items = []
        for item in raw_items:
            # For GRPO, the 'prompt' is the input to the model, and 'answer' is the ground truth for the reward function
            formatted_items.append({
                'prompt': [
                    {'role': 'system', 'content': self.system_prompt_answering},
                    {'role': 'user', 'content': item['question']}
                ],
                'answer': item['answer']
            })
        
        print(f"Created {len(formatted_items)} GRPO training samples for answering.")
        return Dataset.from_list(formatted_items)
        
    def load_dataset(self):
        """Load and format the dataset for GRPO."""
        raw_items = self._load_raw_dataset()
        self.dataset = self._format_grpo_dataset(raw_items)

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer."""
        print(f"Loading model: {self.args.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def train_grpo(self):
        """Train using Group Relative Policy Optimization."""
        self.load_dataset()
        self.setup_model_and_tokenizer()

        peft_config = LoraConfig(
            r=self.args.lora_r, lora_alpha=self.args.lora_alpha, lora_dropout=0.1,
            bias="none", task_type="CAUSAL_LM", target_modules="all-linear"
        )

        training_args = GRPOConfig(
            output_dir=self.args.output_dir,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            logging_steps=1,
            bf16=True,
            max_prompt_length=self.args.max_prompt_length,
            max_completion_length=self.args.max_seq_length - self.args.max_prompt_length,
            save_strategy="epoch",
            use_vllm=True,
            vllm_gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            report_to="wandb" if not self.args.disable_wandb else "none",
        )
        
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[self._combined_reward_func],
            args=training_args,
            train_dataset=self.dataset,
            peft_config=peft_config,
        )
        
        print("Starting GRPO training for A-Agent...")
        self.trainer.train()
        print("A-Agent GRPO training completed.")

    def _combined_reward_func(self, prompts, completions, answer, **kwargs) -> List[float]:
        """Combined reward: format correctness + answer correctness."""
        format_scores = self._format_reward_func(completions)
        correctness_scores = self._correctness_reward_func(completions, answer)
        
        final_rewards = []
        for i in range(len(completions)):
            # Reward is higher if both format and answer are correct
            total_reward = format_scores[i] + correctness_scores[i]
            final_rewards.append(total_reward)
        return final_rewards
    
    def _format_reward_func(self, completions, **kwargs) -> List[float]:
        """Reward: 1.0 if format is correct, 0.0 otherwise."""
        pattern = re.compile(f"(?s)^{self.reasoning_start}.*?{self.reasoning_end}\\s*{self.solution_start}.*?{self.solution_end}$")
        responses = [comp[0]["content"] for comp in completions]
        return [1.0 if pattern.fullmatch(r.strip()) else 0.0 for r in responses]

    def _correctness_reward_func(self, completions, answer, **kwargs) -> List[float]:
        """Reward: 2.0 if extracted answer matches ground truth, 0.0 otherwise."""
        responses = [comp[0]['content'] for comp in completions]
        extracted_answers = [self.extract_xml_answer(r) for r in responses]
        return [2.0 if pred == true_ans else 0.0 for pred, true_ans in zip(extracted_answers, answer)]
    
    def extract_xml_answer(self, text: str) -> str:
        """Extracts the letter from between <answer> tags."""
        try:
            match = re.search(f"{self.solution_start}(.*?){self.solution_end}", text, re.DOTALL)
            if match:
                return match.group(1).strip().upper()
        except Exception:
            pass
        return ""
    
    def setup_inference_model(self):
        """Setup model once for inference mode."""
        if hasattr(self, '_inference_model') and self._inference_model is not None:
            return  # Already loaded
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Setting up inference model on device: {device}")
            
            # Load base model
            print(f"Loading base model: {self.args.model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load tokenizer
            print(f"Loading tokenizer for: {self.args.model_name}")
            self._inference_tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
            if self._inference_tokenizer.pad_token is None:
                self._inference_tokenizer.pad_token = self._inference_tokenizer.eos_token
            self._inference_tokenizer.model_max_length = self.args.max_seq_length
            
            # Find and load checkpoint
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path is None:
                print(f"No trained checkpoints found in {self.args.output_dir}")
                self._inference_model = base_model
            else:
                print(f"Loading LoRA adapters from: {checkpoint_path}")
                self._inference_model = PeftModel.from_pretrained(base_model, checkpoint_path)
            
            self._inference_model = self._inference_model.eval()
            
            print("Inference model setup completed successfully")
            
        except Exception as e:
            error_message = f"Error setting up inference model: {str(e)}"
            print(error_message)
            raise

    def generate_response(self, prompt: str, sys_prompt: str = None) -> str:
        """
        Generate response using the trained model checkpoints.
        
        Args:
            prompt (str): The user question/prompt
            sys_prompt (str, optional): System prompt. If None, uses default.
        
        Returns:
            str: Generated response from the model
        """
        if sys_prompt is None:
            sys_prompt = self.system_prompt_questioning

        try:
            # Setup model if not already done
            self.setup_inference_model()
            
            # Prepare messages
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            full_prompt_text = self._inference_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            # Generate response
            inputs = self._inference_tokenizer(full_prompt_text, return_tensors="pt").to(self._inference_model.device)
            input_len = inputs["input_ids"].shape[-1]
            
            start_time = time.time()
            with torch.inference_mode():
                generation_output = self._inference_model.generate(
                    **inputs, 
                    max_new_tokens=768, 
                    do_sample=False, # For deterministic output
                    pad_token_id=self._inference_tokenizer.eos_token_id # Important for generation
                )
            end_time = time.time()
            
            generated_tokens = generation_output[0][input_len:]
            decoded_response = self._inference_tokenizer.decode(generated_tokens, skip_special_tokens=True)
           
            duration = end_time - start_time
            print(f"Response generated in {duration:.2f}s. Length: {len(decoded_response)} chars.")
            
            return decoded_response.strip()
            
        except Exception as e:
            error_message = f"Error during inference: {str(e)}"
            print(error_message)
            return f"ERROR: {error_message}"
    
    def batch_inference(self):
        """Process multiple questions from a file and save responses."""
        test_file = self.args.test_file or self.args.dataset_file
        output_file = self.args.inference_output
        
        print("="*60)
        print("RUNNING BATCH INFERENCE")
        print("="*60)
        print(f"Loading questions from: {test_file}")
        
        # Load test dataset from JSON
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
            # Handle both array format and single object format
            if isinstance(data, list):
                question_objects = data
            else:
                question_objects = [data]
        
            questions = []
            answers = []
        
            for question_obj in question_objects:
                try:
                    # Extract question text
                    question_text = question_obj.get("question", "")
                
                    # Extract choices and format them as part of the question
                    choices = question_obj.get("choices", [])
                    if choices:
                        # Combine question with choices
                        full_question = question_text + "\n" + "\n".join(choices)
                    else:
                        full_question = question_text
                
                    # Extract answer
                    answer_letter = question_obj.get("answer", "").strip()
                
                    if full_question and answer_letter:
                        questions.append(full_question)
                        answers.append(answer_letter)
                    else:
                        print(f"Warning: Incomplete question data in object: {question_obj}")
                    
                except Exception as e:
                    print(f"Warning: Error processing question object: {e}")
                    continue
        
            print(f"Successfully loaded {len(questions)} questions from JSON file")
        
        except FileNotFoundError:
            print(f"Error: Test file {test_file} not found.")
            return
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in {test_file}: {e}")
            return
        except Exception as e:
            print(f"Error loading test file: {e}")
            return

        if not questions:
            print("No valid questions found in test file.")
            return

        print(f"Processing {len(questions)} questions...")
        
        # Setup inference model once before processing all questions
        print("Setting up inference model...")
        try:
            self.setup_inference_model()
        except Exception as e:
            print(f"Failed to setup inference model: {e}")
            return

        with open(output_file, "w", encoding='utf-8') as f:
            f.write("# Batch Inference Results\n\n")
            f.write(f"Model: {self.args.model_name}\n")
            f.write(f"Training Type: {self.args.training_type}\n")
            f.write(f"Checkpoint: {self.find_latest_checkpoint()}\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            correct_count = 0
            format_correct_count = 0  # Count of responses with extractable answers
            total_count = len(questions)
            
            # Process all questions with the same loaded model
            for idx, (question, expected_answer) in enumerate(zip(questions, answers), 1):
                print(f"Processing question {idx}/{total_count}...")

                response = self.generate_response(question, sys_prompt=self.system_prompt_questioning if self.args.training_type == 'sft' else self.system_prompt_answering)
                extracted_answer = self.extract_xml_answer(response)
                
                # Check if answer was extractable (format correctness)
                is_format_correct = extracted_answer != ""
                if is_format_correct:
                    format_correct_count += 1
                
                # Check if answer is correct
                is_correct = extracted_answer == expected_answer
                if is_correct:
                    correct_count += 1
            
                f.write(f"## Question {idx}\n\n")
                f.write(f"**Question:** {question}\n\n")
                f.write(f"**Expected Answer:** {expected_answer}\n\n")
                f.write(f"**Model Response:**\n```\n{response}\n```\n\n")
                f.write(f"**Extracted Answer:** {extracted_answer if extracted_answer else 'N/A (Format Error)'}\n\n")
                f.write(f"**Format Correct:** {'✅' if is_format_correct else '❌'}\n\n")
                f.write(f"**Answer Correct:** {'✅' if is_correct else '❌'}\n\n")
                f.write("---\n\n")

            # Calculate percentages
            accuracy = correct_count / total_count * 100
            format_accuracy = format_correct_count / total_count * 100
            
            # Write summary
            f.write(f"## Summary\n\n")
            f.write(f"Total Questions: {total_count}\n")
            f.write(f"Correct Answers: {correct_count}\n")
            f.write(f"Format Correct: {format_correct_count}\n")
            f.write(f"Answer Accuracy: {accuracy:.2f}%\n")
            f.write(f"Format Accuracy: {format_accuracy:.2f}%\n")
            
        print(f"Batch inference complete. Results saved to {output_file}")
        print(f"Answer Accuracy: {correct_count}/{total_count} ({accuracy:.2f}%)")
        print(f"Format Accuracy: {format_correct_count}/{total_count} ({format_accuracy:.2f}%)")
            
    def cleanup_inference_model(self):
        """Clean up inference model to free memory."""
        if hasattr(self, '_inference_model') and self._inference_model is not None:
            del self._inference_model
            self._inference_model = None
        if hasattr(self, '_inference_tokenizer') and self._inference_tokenizer is not None:
            del self._inference_tokenizer
            self._inference_tokenizer = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Inference model cleaned up")

    def test_single_inference(self):
        """Test inference with a single question."""
        if not hasattr(self.args, 'test_question') or not self.args.test_question:
            print("No test question provided. Use --test_question argument.")
            return
        
        print("="*60)
        print("RUNNING SINGLE INFERENCE TEST")
        print("="*60)
        print(f"Question: {self.args.test_question}")
        
        # Setup inference model
        print("Setting up inference model...")
        try:
            self.setup_inference_model()
        except Exception as e:
            print(f"Failed to setup inference model: {e}")
            return
        
        # Generate response
        print("Generating response...")
        response = self.generate_response(self.args.test_question, sys_prompt=self.system_prompt_questioning if self.args.training_type == 'sft' else self.system_prompt_answering)
        
        extracted_answer = self.extract_xml_answer(response)
        
        # Display results
        print("\n" + "="*60)
        print("INFERENCE RESULTS")
        print("="*60)
        print(f"Question: {self.args.test_question}")
        print(f"\nModel Response:\n{response}")
        print(f"\nExtracted Answer: {extracted_answer}")
        print("="*60)
        
        # Save results if output file specified
        if hasattr(self.args, 'inference_output') and self.args.inference_output:
            with open(self.args.inference_output, "w", encoding='utf-8') as f:
                f.write("# Single Inference Test Results\n\n")
                f.write(f"Model: {self.args.model_name}\n")
                f.write(f"Training Type: {self.args.training_type}\n")
                f.write(f"Checkpoint: {self.find_latest_checkpoint()}\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Question:** {self.args.test_question}\n\n")
                f.write(f"**Model Response:**\n```\n{response}\n```\n\n")
                f.write(f"**Extracted Answer:** {extracted_answer}\n")
            
            print(f"Results saved to: {self.args.inference_output}")

    # ... (Keep inference methods like setup_inference_model, generate_response, etc., from the original class)
    # ... (They will work as is, just pass an aptitude question as the prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A-Agent GRPO Trainer")
    # General
    parser.add_argument("--model_name", type=str, default="/jupyter-tutorial/hf_models/Llama-3.2-1B-Instruct", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="checkpoints/a_agent/grpo", help="Output directory")
    parser.add_argument("--dataset_file", type=str, default="aptitude_questions.json", help="Path to dataset")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs")
    # Training
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train")
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.8)
    # WandB
    parser.add_argument("--disable_wandb", action="store_true")
    # ... (add other necessary args)
    
    args = parser.parse_args()
    
    trainer = AnswerAgentTrainer(args)
    if args.mode == 'train':
        trainer.train_grpo()
    # Add inference logic if needed