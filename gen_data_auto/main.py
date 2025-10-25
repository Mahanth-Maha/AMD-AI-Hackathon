import openai, json
openai.api_key = "maha-123"
openai.base_url = "http://localhost:8000/v1"

def seatting_arrangement_que_gen(topic: str) -> str:
    prompt = f"""
You are an expert logical reasoning tutor trained to create challenging and realistic logic puzzles in English involving family trees, generations, and relationships.

Your task is to generate one multiple-choice puzzle under the given topic, and return the result as a valid JSON object.

Only output the JSON, and ensure that:
- The structure matches the required schema
- All field values are properly escaped and quoted
- Each `choices` item starts with one of: "A) ", "B) ", "C) ", or "D) "
- `answer` is a single uppercase letter: "A", "B", "C", or "D"
- `explanation` gives concise step-by-step reasoning (<100 words)

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

✅ EXAMPLE 1:
{{
"topic": "Puzzles involving generations and family tree logic",
"question": "A is B's mother. B is C's only son. C is D's husband. D's only sibling, E, is unmarried. E's father, F, has only one son. G is F's wife. If C has no siblings, how is G's son-in-law related to A?",
"choices": [
"A) Husband",
"B) Son",
"C) Brother",
"D) Father"
],
"answer": "A",
"explanation": "A is B's mother, and B is C's only son. This implies that A and C are B's parents."
}}

✅ EXAMPLE 2:
{{
"topic": "Puzzles involving generations and family tree logic",
"question": "P's father, Q, is the only son of R's husband. R has only one child. S is R's daughter-in-law. S has an only child, T. If P is male and T is female, how is P's wife related to T's paternal grandmother?",
"choices": [
"A) Daughter",
"B) Niece",
"C) Granddaughter-in-law",
"D) Sister-in-law"
],
"answer": "C",
"explanation": "Q is R's son, so R is P's grandmother. P's wife is granddaughter-in-law of R."
}}

✅ Now generate a new question under the same topic and return only the JSON object, nothing else.
"""

    return prompt.strip()



topic = "Seating Arrangement"
prompt = seatting_arrangement_que_gen(topic)

response = openai.ChatCompletion.create(
    model="llama3-70b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.5,
    max_tokens=1024
)

content = response.choices[0].message.content.strip()

# Auto-clean if needed
content = content.replace("```json", "").replace("```", "").strip()

try:
    data = json.loads(content)
    print(json.dumps(data, indent=2))
except Exception as e:
    try:
        # save it to txt file for debugging
        with open("debug_output.txt", "w") as f:
            f.write(content)
    except Exception as file_error:
        print("⚠️ Failed to save debug output:", file_error)
    finally:
        print("⚠️ Invalid JSON:\n", content)
        
        