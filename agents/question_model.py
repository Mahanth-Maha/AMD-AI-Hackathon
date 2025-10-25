# Starting with Qwen3-4B
import time
from typing import Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
import re
import time
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QAgent(object):
    def __init__(self, model_name="/jupyter-tutorial/hf_models/Qwen3-4B", **kwargs):
        self.model_type = kwargs.get('model_type', '4B').strip()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            padding_side='left'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def _clean_output(self, content: str) -> str:
        # Remove code blocks/backticks
        content = re.sub(r"```(?:json)?|```", "", content)
        # Replace malformed choice labels like "A) A)" → "A)"
        content = re.sub(r'([A-D])\)\s*\1\)', r'\1)', content)
        # Strip and collapse
        return content.strip()

    def _safe_json_parse(self, content: str) -> Optional[dict]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    def generate_response(self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs):
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        if isinstance(message, str):
            message = [message]

        # Prepare all messages for batch processing
        all_messages = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": msg}]
            for msg in message
        ]

        texts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for m in all_messages
        ]

        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        tgps_show_var = kwargs.get('tgps_show', False)
        if tgps_show_var:
            start_time = time.time()

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=kwargs.get('max_new_tokens', 1024),
            pad_token_id=self.tokenizer.pad_token_id,
        )

        if tgps_show_var:
            generation_time = time.time() - start_time

        batch_outs = []
        if tgps_show_var:
            token_len = 0

        for input_ids, generated_sequence in zip(model_inputs.input_ids, generated_ids):
            output_ids = generated_sequence[len(input_ids):].tolist()
            if tgps_show_var:
                token_len += len(output_ids)

            # Remove <|im_end|> or similar EOS token tail if present
            eos_token = 151668
            index = len(output_ids) - output_ids[::-1].index(eos_token) if eos_token in output_ids else 0
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
            cleaned = self._clean_output(content)

            # Attempt safe JSON parse
            parsed = self._safe_json_parse(cleaned)
            if parsed is None:
                # Try fixing trailing commas, newlines etc.
                cleaned = re.sub(r",\s*}", "}", cleaned)
                cleaned = re.sub(r",\s*]", "]", cleaned)
                parsed = self._safe_json_parse(cleaned)

            batch_outs.append(parsed if parsed else cleaned)

        if tgps_show_var:
            return batch_outs[0] if len(batch_outs) == 1 else batch_outs, token_len, generation_time

        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None


if __name__ == "__main__":
    # Single example generation
    model = QAgent()
    prompt = f"""
    Question: Generate a hard MCQ based question as well as their 4 choices and its answers on the topic, Number Series.
    Return your response as a valid JSON object with this exact structure:

        {{
            "topic": Your Topic,
            "question": "Your question here ending with a question mark?",
            "choices": [
                "A) First option",
                "B) Second option", 
                "C) Third option",
                "D) Fourth option"
            ],
            "answer": "A",
            "explanation": "Brief explanation of why the correct answer is right and why distractors are wrong"
        }}
    """
    
    response, tl, tm = model.generate_response(prompt, tgps_show=True, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True)
    print("Single example response:")
    print("Response: ", response)
    print(f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec")
    print("+-------------------------------------------------\n\n")

    # Multi example generation
    prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, tm = model.generate_response(prompts, tgps_show=True, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True)
    print("\nMulti example responses:")
    for i, resp in enumerate(responses):
        print(f"Response {i+1}: {resp}")
    print(f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec")
