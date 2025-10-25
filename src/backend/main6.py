import datetime
from difflib import context_diff
from numpy import save
import requests
import json
import datetime

from tqdm import tqdm 

start_time = datetime.datetime.now()

def truth_teller_and_liar_batch_prompt(topic: str, seed_context: str) -> str:
    return f"""
You are an expert logical reasoning tutor. Your task is to generate **3 realistic, self-contained multiple-choice logic puzzles** under the topic: "{topic}".

These puzzles must involve Truth-teller and Liar Problems  and **include enough context within the question** to be solvable. Do **not** rely on external knowledge or assumptions — the question itself must provide everything needed to deduce the answer.

---

🎯 **Rules:**
- Return a **JSON array of 5 question objects**, each following the format below.
- Do **not** include any explanation text before/after the JSON.
- Each puzzle must be **answerable from its own question**.
- Each puzzle must be simple and se lf-contained.
- Every `question` item must be within 90 words (<= 90 words).
- Every `choices` item must start with `"A) "`, `"B) "`, etc.
- `answer` must be one of: `"A"`, `"B"`, `"C"`, `"D"`.
- `explanation` must briefly explain the reasoning (<500 words).
- Use or build around this context if needed, as this is a tutorial to frame the questions:
  <CONTEXT>
  {seed_context}
  </CONTEXT>

<SAMPLE>
{{
    "question": "On an island, three inhabitants A, B, and C make statements: A says, 'B is a liar.' B says, 'C is a liar.' C says, 'A and B are of different types.' If exactly two are liars, who is the truth-teller?",
    "choices": [
        "A) A",
        "B) B",
        "C) C",
        "D) None"
    ],
    "answer": "C",
    "explanation": "C's statement is consistent with A and B being liars. If A and B are both liars, then they are the same type, making C's statement false—but C is supposed to be the truth-teller. The only valid configuration with exactly two liars is A and B lying, and C truthfully stating A and B are of different types, which is false—so contradiction. Try C as liar: then A and B must differ. Let A be truthful (B is liar), then B lies (C is liar), which fits. So only consistent solution: A = truth-teller."
}},
{{
    "question": "In a realm with knights (always truthful), knaves (always lie), and spies (can do either), X says: 'If I am a knight, then Y is a spy.' Y says: 'X is a knave.' What is X and Y?",
    "choices": [
        "A) X: knight, Y: spy",
        "B) X: knave, Y: knight",
        "C) X: spy, Y: knave",
        "D) X: knave, Y: spy"
    ],
    "answer": "D",
    "explanation": "Assume X is a knight -> Y must be a spy (from X's statement). Then Y says 'X is a knave'—which contradicts Y being a spy and telling truth. Now try X as knave: knaves lie, so the implication must be false. But 'If P then Q' is only false when P is true and Q is false, which doesn't apply here—so statement vacuously true. Y as spy saying 'X is a knave' may be true or false. Only consistent config: X is knave (lying), Y is spy (can say anything)."
}},
{{
    "question": "Person A says, 'B is a liar.' Person B says, 'A is a truth-teller.' Who is the truth-teller and who is the liar?",
    "choices": [
        "A) Both A and B are truth-tellers",
        "B) Both A and B are liars",
        "C) A is a truth-teller, B is a liar",
        "D) A is a liar, B is a truth-teller"
    ],
    "answer": "D",
    "explanation": "If A were a truth-teller, then B would be a liar. But B says A is a truth-teller, which would be a lie if B were a liar, meaning A is a liar—a contradiction. Therefore, A must be a liar, so B is not a liar (i.e., B is a truth-teller). B's statement that A is a truth-teller is false, which fits because A is a liar."
}}
</SAMPLE>

---

📦 **FORMAT of each item:**
{{
  "topic": "{topic}",
  "question": "<Question including all relevant context>",
  "choices": [
    "A) ...",
    "B) ...",
    "C) ...",
    "D) ..."
  ],
  "answer": "A",
  "explanation": "<Explain why this is correct>"
}}

✅ Return only the JSON array of 3 objects. No markdown, no commentary.
"""


def call_vllm_chat(prompt: str):
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Authorization": "Bearer maha-123",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.75,
        "max_tokens": 1024
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    # print(result)
    try:
        raw_output = result["choices"][0]["message"]["content"]
        cleaned = raw_output.replace("```json", "").replace("```", "").strip()
    except (KeyError, IndexError):
        print("⚠️ Invalid response format: \nresult::", result)
        return None

    try:
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError:
        print("❌ Failed to parse:", raw_output)
        return None

def get_random_txt_file_from_dir(directory: str) -> str:
    import os
    import random

    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    if not txt_files:
        return ""

    random_file = random.choice(txt_files)
    return os.path.join(directory, random_file)

def load_from_txt_file(file_path: str) -> str:
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"⚠️ File not found: {file_path}")
        return ""
    except Exception as e:
        print(f"⚠️ Error reading file {file_path}: {e}")
        return ""
        
  
# Automatically generate and save N batches incrementally to new JSON files
N = 2
topic = "Truth Teller and Liar Problems"

delta = 0
def get_textfiles_and_extract_delta():
    import os
    import re

    txt_files = [f for f in os.listdir(".") if f.endswith('.txt')]
    if not txt_files:
        return 0

    # Extract the highest number from filenames like "tl_dataset_1.json"
    numbers = []
    for filename in txt_files:
        match = re.search(r'tl_dataset_(\d+)\.json', filename)
        if match:
            numbers.append(int(match.group(1)))

    return max(numbers) if numbers else 0
delta = get_textfiles_and_extract_delta()


for i in tqdm(range(N), desc="Generating batches"):
    context_file = get_random_txt_file_from_dir(".")
    context = load_from_txt_file(context_file)
    prompt = truth_teller_and_liar_batch_prompt(topic, context)
    result = call_vllm_chat(prompt)

    output_filename = f"tl_dataset_{i+ 1 + delta}.json"
    if result:
        print(f"✅ Batch {i+1}: Saving to {output_filename}")
        try:
            with open(output_filename, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save to {output_filename}: {e}")
    else:
        print(f"⚠️ Batch {i+1}: No valid result returned from the model.")

end_time = datetime.datetime.now()

print(f"✅ Finished generating questions in {end_time - start_time} seconds")