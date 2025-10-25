import os
import random
import argparse
import json
import time
import re
from typing import List, Dict, Optional
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
import torch
import wandb

class QuestionGeneratorTrainer:
    """
    Trainer class for fine-tuning a model to GENERATE aptitude questions using SFT.
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
        
        # --- XML Tags for Question Generation ---
        self.topic_start = "<topic>"
        self.topic_end = "</topic>"
        self.question_start = "<question>"
        self.question_end = "</question>"
        self.choices_A_start = "<choiceA>"
        self.choices_A_end = "</choiceA>"
        self.choices_B_start = "<choiceB>"
        self.choices_B_end = "</choiceB>"
        self.choices_C_start = "<choiceC>"
        self.choices_C_end = "</choiceC>"
        self.choices_D_start = "<choiceD>"
        self.choices_D_end = "</choiceD>"
        self.answer_start = "<answer>"
        self.answer_end = "</answer>"
        self.explanation_start = "<explanation>"
        self.explanation_end = "</explanation>"

        self.system_prompt_questioning = f"""
        You are an expert in logical reasoning and complex problem-solving.
        Your task is to generate high-quality multiple-choice questions (MCQs) on a given topic.
        The topics can be "Truth-teller and Liar Problems", "Seating Arrangements (Linear, Circular)", or "Puzzles involving generations and family tree logic".
        You must strictly follow the provided format using XML-like tags.
        Each question must be challenging, self-contained, and have a clear question, four distinct options, the correct answer letter, and a detailed explanation.

        The required format is:
        {self.topic_start}{{topic}}{self.topic_end}
        {self.question_start}{{question}}{self.question_end}
        {self.choices_A_start}{{choice_a}}{self.choices_A_end}
        {self.choices_B_start}{{choice_b}}{self.choices_B_end}
        {self.choices_C_start}{{choice_c}}{self.choices_C_end}
        {self.choices_D_start}{{choice_d}}{self.choices_D_end}
        {self.answer_start}{{answer}}{self.answer_end}
        {self.explanation_start}{{explanation}}{self.explanation_end}

        Do not include any text outside of these tags.
        """
        self._setup_environment()
        self._display_config()

    def _setup_environment(self):
        """Setup environment variables and GPU configuration."""
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", self.args.gpu_ids)
        print(f"PyTorch detected number of available devices: {torch.cuda.device_count()}")

    def _display_config(self):
        """Display training configuration."""
        print("=" * 60)
        print("Q-AGENT (QUESTION GENERATOR) SFT TRAINER")
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
            
            topics = ["Puzzles involving generations and family tree logic", "Truth-teller and Liar Problems", "Seating Arrangements (Linear, Circular)"]
            for idx, q_obj in enumerate(question_objects, 1):
                topic = q_obj.get("topic", random.choice(topics))
                question = q_obj.get("question")
                choices = q_obj.get("choices", [])
                answer = q_obj.get("answer")
                explanation = q_obj.get("explanation")
                
                if all([topic, question, len(choices) == 4, answer, explanation]):
                    items.append({
                        "topic": topic,
                        "question": question,
                        "choices": choices,
                        "answer": answer,
                        "explanation": explanation,
                    })
                else:
                    print(f"Warning: Skipping incomplete question data at index {idx}")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        
        print(f"Successfully loaded {len(items)} questions for training.")
        return items

    def _format_sft_dataset(self, raw_items: List[Dict[str, str]]) -> Dataset:
        """Format raw items for SFT training to generate questions."""
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        formatted_items = []
        for item in raw_items:
            # The model's expected output (the assistant's message)
            model_completion = (
                f"{self.topic_start}{item['topic']}{self.topic_end}\n"
                f"{self.question_start}{item['question']}{self.question_end}\n"
                f"{self.choices_A_start}{item['choices'][0]}{self.choices_A_end}\n"
                f"{self.choices_B_start}{item['choices'][1]}{self.choices_B_end}\n"
                f"{self.choices_C_start}{item['choices'][2]}{self.choices_C_end}\n"
                f"{self.choices_D_start}{item['choices'][3]}{self.choices_D_end}\n"
                f"{self.answer_start}{item['answer']}{self.answer_end}\n"
                f"{self.explanation_start}{item['explanation']}{self.explanation_end}"
            )
            
            # The user prompt that triggers the generation
            user_prompt = f"Generate a new multiple-choice question on the topic of '{item['topic']}'."
            
            chat_messages = [
                {'role': 'system', 'content': self.system_prompt_questioning},
                {'role': 'user', 'content': user_prompt},
                {'role': 'assistant', 'content': model_completion}
            ]
            
            formatted_text = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=False
            )
            formatted_items.append({"text": formatted_text})
        
        print(f"Created {len(formatted_items)} SFT training samples for question generation.")
        return Dataset.from_list(formatted_items)

    def load_dataset(self):
        """Load and format the dataset for SFT."""
        raw_items = self._load_raw_dataset()
        self.dataset = self._format_sft_dataset(raw_items)

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer."""
        print(f"Loading model: {self.args.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name, 
            torch_dtype=torch.bfloat16, 
            # device_map="auto", 
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def train(self):
        """Main SFT training function."""
        self.load_dataset()
        self.setup_model_and_tokenizer()

        peft_config = LoraConfig(
            r=self.args.lora_r, lora_alpha=self.args.lora_alpha, lora_dropout=0.1,
            bias="none", task_type="CAUSAL_LM", target_modules="all-linear"
        )

        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            lr_scheduler_type="cosine",
            report_to="wandb" if not self.args.disable_wandb else "none",
        )

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=training_args,
            train_dataset=self.dataset,
            peft_config=peft_config,
            # dataset_text_field="text",
            # max_seq_length=self.args.max_seq_length,
        )

        print("Starting SFT training for Q-Agent...")
        self.trainer.train()
        self.trainer.save_model(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
        print("Q-Agent SFT training completed.")
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
                # device_map="auto",
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
    # ... (They will work as is, just remember to use a prompt like "Generate a question about X")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Agent SFT Trainer")
    # General
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Model name")
    parser.add_argument("--output_dir", type=str, default="checkpoints/q_agent/sft", help="Output directory")
    parser.add_argument("--dataset_file", type=str, default="aptitude_questions.json", help="Path to dataset")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs")
    # Training
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    # WandB
    parser.add_argument("--disable_wandb", action="store_true")
    # ... (add other necessary args)
    
    args = parser.parse_args()
    
    trainer = QuestionGeneratorTrainer(args)
    if args.mode == 'train':
        trainer.train()
    # Add inference logic if needed