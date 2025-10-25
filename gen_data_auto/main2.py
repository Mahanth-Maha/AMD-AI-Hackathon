from difflib import context_diff
from numpy import save
import requests
import json

def seatting_arrangement_que_gen(topic: str, context: str) -> str:
    return f"""
You are an expert logical reasoning tutor trained to create challenging and realistic logic puzzles in English involving family trees, generations, and relationships.

Your task is to generate one multiple-choice puzzle under the given topic and give context to generate from, and return the result as a valid JSON object.



Only output the JSON, and ensure that:
- The structure matches the required schema
- All field values are properly escaped and quoted
- Each `choices` item starts with one of: "A) ", "B) ", "C) ", or "D) "
- `answer` is a single uppercase letter: "A", "B", "C", or "D"
- `explanation` gives concise step-by-step reasoning (<100 words)
- The Question generated should have enough context to answer it. This should not be violated and taken at most care that the context in question is enough
- `answer` and `explanation` should be logically infered by the question itself, try to answer the question once and check 

Output format:
{{
  "topic": "{topic}",
  "question": "<Your question here>",
  "choices": [
    "A) ...",
    "B) ...",
    "C) ...",
    "D) ..."
  ],
  "answer": "A",
  "explanation": "<Your explanation here>"
}}

Do NOT return anything other than the pure JSON object.

--- 
context start
<CONTEXT>
{context}
</CONTEXT>
context end
---

✅ Topic: "{topic}"

✅ FORMAT:
```json
{{
  "topic": "<repeat the topic here>",
  "question": "<descriptive question text, 1-2 sentences>",
  "choices": [
    "A) ...",
    "B) ...",
    "C) ...",
    "D) ..."
  ],
  "answer": "B",
  "explanation": "Brief logical reasoning (max 100 words)"
}}
```
✅ RULES:

Do not add any markdown like triple backticks or headings

Do not wrap the JSON in any text or commentary

Do not write "Here is the JSON:"

Output must be parsable directly by json.loads()

✅ Now generate a new question under the same topic and return only the JSON object, nothing else.
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

# Example usage
topic = "Seating Arrangement"
context_file = get_random_txt_file_from_dir(".")
context = load_from_txt_file(context_file) 


prompt = seatting_arrangement_que_gen(topic, context)
result = call_vllm_chat(prompt)


append_to_json = 'sa_dataset.json'
def save_to_json(data, filename):
    try:
        with open(filename, 'a') as f:
            json.dump(data, f)
            f.write('\n')
    except Exception as e:
        print(f"⚠️ Failed to save to {filename}: {e}")

if result:
    print(json.dumps(result, indent=2))
    save_to_json(result, append_to_json)
else :
    print("⚠️ No valid result returned from the model.")
    

