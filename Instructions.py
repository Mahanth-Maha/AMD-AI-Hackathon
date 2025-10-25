# %% [markdown]
# <h1 align='center'>Welcome to the AMD AI Premier League (AAIPL)!</h1>
# 

# %% [markdown]
# 
# ---
# ## Task:
# Here you will be building:
# 1.  A question agent or Q-agent (e.g., [question_model.py](./agents/question_model.py) & [question_agent.py](./agents/question_agent.py)) that will ask some $N$ puzzle-based questions based on some given [topics](./assets/topics.json). *Note your question agent should output questions in the format specified in [sample_question.json](./assets/sample_question.json)*.
# 2.  Also, an answer agent or A-agent (e.g., [answer_model.py](./agents/answer_model.py) & [answer_agent.py](./agents/answer_agent.py)) that answer questions asked from question agent. Here too, the format of the output should follow as specified in [sample_answer.json](./assets/sample_answer.json) file.
# 
# ---
# 

# %% [markdown]
# ## Instructions
# 
# 1.  How to initiate your work station?
#     -   Type `http://xxx.xxx.xxx.xxx:8080` link (in *Chrome*), where `xxx.xxx.xxx.xxx` is the 12 (or 11) digit IP shared with each team. Hiting this URL, will open up a Jupyter lab page. Enter the token as `amdhack` (same for all).
#     -   Upon landing into Jupyter Lab page, on left side (i.e., folders), you will see `AAIPL/`, `hf_models/`. All this will be inside `/jupyter-tutorial` directory.
#     -   Within `hf_models/` there are some base checkpoints which will be used for question and answer agent creation. For both, Q-agent and A-agent, we are using `Qwen3-4B` (with `enable_thinking=False` to avoid thinking tokens) as base model.
#     -   Note that inside `AAIPL/`, there is a `tutorial` folder which consists of python scripts and `.ipynb` file demonstrating how to do *SFT*, *GRPO*, and *Prompt-tuning*. You are encouraged to, of course, improve/edit upon this (BYOC). No need to stick with this strictly.
# 2.  You got 24 hrs to bake this cake!
# 3.  All the BEST!

# %% [markdown]
# ## 📚 Table of Contents:
# - 🏁 [Welcome to the AMD AI Premier League (AAIPL)!](#welcome-to-the-amd-ai-premier-league-aaipl)
# - 📝 [Task](#task)
# - ⚙️ [Instructions](#instructions)
# - 🏏 [Tournament Overview](#tournament-overview)
# - 📋 [Guidelines](#guidelines)
#     - [Naming Conventions](#naming-conventions)
#     - [Format Overview](#format-overview)
# - 🛠️ [What you will submit?](#️what-you-will-submit)
# - ⚠️ [RESTRICTIONS](#restrictions)
#     - [ALLOWED](#allowed)
# - 📂 [Directory & Files overview](#directory--files-overview)
# - 🚀 [Env Setup](#env-setup)
# - 🎮 [Let the GAME begin!!!](#let-the-game-begin)
#     - 🤔 [Q-Agent](#q-agent)
#         - ✅ [Basic format-checks for questions from Q-agent](#basic-format-checks-for-questions-from-q-agent)
#     - 🤖 [A-agent](#a-agent)
#         - ✅ [Basic format-checks for answers from A-agent](#basic-format-checks-for-answers-from-a-agent)
# - 🏅 [Evaluation](#evaluation)
#     - 📊 [Scoring Criteria](#scoring-criteria)
# - ⏱ [Time Limit](#time-limit)
# <!-- - 🏆 [LeaderBoard UI/UX](#leaderboard-uiux) -->

# %% [markdown]
# ## Tournament Overview
# <!-- 🏏  -->
# 1.  All matches in this tournament will be **1v1** knockout format where two teams, Team-A vs Team-B, will compete with their Q-agent (question agent) and A-agent (answer agent). For simplicity think Q-agent to be bowler and A-agent to be batsman.
# 2.  Like a cricket match, this would also have two innings:
# 
#     -   1st inning:
#         *   $N$ Question from the Q-agent (Team-A) and their corresponding $N$ answers from the A-agent (Team-B).
#         *   Q-agent score (Team-A): Say, $40$
#         *   A-agent score (Team-B): $60$
# 
#     -   2nd inning:
#         *   $N$ Question from the Q-agent (Team-B) and their respective $N$ responses from the A-agent (Team-A).
#         *   Q-agent score (Team-B): Say, $70$
#         *   A-agent score (Team-A): $30$
#     -   Final Score:
#         *   Team-A score $=$ 1st inning Q-agent score $+$ 2nd inning A-agent score $= 40 + 30 = 70$
#         *   Team-B score $=$ 1st inning Q-agent score $+$ 2nd inning A-agent score $= 60 + 70 = 130$
# 
#     -   Winner: **Team-B** with a score of $130$.
#     -   For more info on <b> how SCORING is done</b>, kindly refer to this [cell](#scoring-criteria).
# 
# <u>NOTE</u>: In case of **TIE**, we will use some (closed) benchmark questions, we will evaluate your answer agents (A-agent) and rank the teams accordingly.
# 
# **Whichever Team's Q-agent fails to generate atleast $50\%$ of `num_questions` (where `num_questions` ranges from $2$ to $1000+$) of the questions correctly (as per [format-checking](#format-overview)) will be automatically disqualified.**<br>
# <u>Note</u>: Here $N$ denotes the number of filtered / format-correct questions.
# 

# %% [markdown]
# 
# ## Guidelines:
# <!-- 📋  -->
# #### Naming Conventions:
# <ol type="a">
#     <li>Rename this whole folder as <code>AAIPL_your_IP</code> if not done already. This <code>your_IP</code> will be <code>_</code> separated IPv4 address, no special-characters allowed. Follow the below <a href="#what-you-will-submit">cell</a> for more info</li>
#     <li> For Q-agent:
#         <ol type="i">
#             <li>For Q-agent wrapper <code>.py</code> file: <code>agents/question_agent.py</code>.</li>
#             <li>For Q-agent model <code>.py</code> file: <code>agents/question_model.py</code>.</li>
#         </ol>
#     </li>
#     <li> For A-agent:
#         <ol type="i">
#             <li>For A-agent wrapper <code>.py</code> file: <code>agents/answer_agent.py</code>.</li>
#             <li>For A-agent model <code>.py</code> file: <code>agents/answer_model.py</code>.</li>
#         </ol>
#     </li>
# </ol>
# 
# 
# #### Format Overview
# -   <u>Q-Agent</u>: Given a topic, the Q-agent should generate questions in the specified JSON format:
#     ```json
#     {
#     "topic": "<Topic of the Question>",
#     "question": "<full question text>",
#     "choices": [
#         "A) <choice A text>",
#         "B) <choice B text>",
#         "C) <choice C text>",
#         "D) <choice D text>"
#     ],
#     "answer": "<correct choice letter only>",
#     "explanation": "brief explanation within 100 words for why the answer is correct"
#     }
#     ```
#     from which we will extract **ONLY** the **"Question"** and **"Choices"** keys and feed it to the answer agent. The **"Topic"**, **"Question"**, **"Choices"**, and **"Answer"** will be verified for correctness from an Oracle.
# -   <u>A-agent</u>: Given a Question and Choices, A-agent should produce answer in the format of:
#     ```json
#     {
#         "answer": "<correct choice letter only>",
#         "reasoning": "brief reasoning within 100 words for why the answer is correct"
#     }
#     ```
#     where we will extract ONLY the **"Answer"** key and compare it with **"Answer"** from the opponent's question.
# -   *<u>Remarks</u>: Having explanation and reasoning is a plus. Not having them doesn't disqualify the question or answer being correct.*
#     
# **<u>Note</u>**: *We will only consider those responses from the Q-agent and the A-agent which follow the above format.*
# 

# %% [markdown]
# ## What you will submit?
# <!-- 🛠️  -->
# You need to submit your code which should contain these main files:
# 
# <!-- *   `q_agent.py` (with one arg as `num_quetions: int`) - On running with `num_questions=20` should generate 20 questions in the required format as above.
# *   `a_agent.py` (with two args as `Question: str` and `Choices: List[str]`) - On running with the above two args should produce the o/p in the required format as above. -->
# <!-- 1. Submit the whole folder AAIPL with its name modified to `AAIP_<your_team_name_in_alphanumeric>`. No special characters, e.g., `#`, `@`, `!`, etc. are allowed in the team name. 
#    - Example: `AAIP_Team1` or `AAIP_Team23` or `AAIP_Win47` are valid, but `AAIP_Team#1` or `AAIP_Team@1` are not.
# 2. Also put Checkpoints (e.g., `model.safetensors` or `.pt` or `.pth`) file (situated at e.g., `ckpts/`) - given that they get successfully loaded automatically, when we execute inference as done above for both, question and answer agent.
# 3. `requirements.txt` - This file lists all the extra dependencies required to run your agents apart from `default_requirements.txt`. -->
# 
# 1. Rename the `AAIPL` folder to `AAIPL_<your_IP_address>` if not done already. NO special characters, e.g., `#`, `@`, `!`, etc. are allowed except underscore, `_`, in the team name. 
#    - Example: `AAIPL_192_154_162_143` or `AAIPL_192_154_182_14` are valid, but `AAIPL_Team#1` or `AAIPL_Team@1` are not. 
# 1. **No need to upload anything to anywhere, we'll collect your codes at sharp 2:00 PM - 13th July, 2025 from your Jupyter Server.**
# 2. <span style="color: red;">Don't forget to add your PPT (in `solution.pdf` format) summarizing the techniques you adopted to execute this task better, relative to this file (i.e., just inside `AAIPL_xxx_xxx_xxx_xxx` folder).</span>
# 3. **ENSURE model checkpoint(s) (e.g., `model.safetensors` or `.pt` or `.pth`) is(are) loading and expected files are getting generated from Q-agent and A-agent, when inference is done. And put all your checkpoints in the `ckpt/` folder, located just inside `AAIPL_<your_IP>/`.**
# 4. **<u>NOTE</u>: You are not required to generate any `.json` for us, we'll do that for you during evaluation setting a specific value to $N$.**
# 
# <u><span style="color: blue">NOTE</span></u>: These files will be checked for any hardcoding, RAG, or other unfair practices.<br>
# <u><span style="color: red">REMARKS / CAUTION</span></u>: A-agent is equally important as Q-agent. So, please do focus on both.

# %% [markdown]
# ## RESTRICTIONS
# <!-- ⚠️ -->
# 
# 1.  Kindly don't use any sort of ***RAG (Retrieval Augmented Generation)*** techniques. If found, the submission won't be considered for further evaluations.
# 2.  **Usage of base models other than what given for Question (i.e., `Qwen3-4B`) and Answer (i.e., again `Qwen3-4B`) agent, will lead to disqualification.**
# 3.  Do follow the guidelines as mentioned in [What you will submit?](#what-you-will-submit) section.
# 4.  **<span style="color: red">NO</span> LAST Minute Submission**: The submission deadline is strict. Upload link will expires just one minute before the deadline. So, please make sure you submit your code well in advance.
# 5.  Any **<span style="color: red">HACKEY</span>** approach or **hard-coding** will lead to disqualification.
#     -   E.g., Any hard-coded *adversarial attacks* that make A-agent hallucinates.
# 6.  **Language Restriction**: ONLY English language is allowed for both Q-agent and A-agent. Any other language will lead to disqualification.
# 7.  Strictly stay within the `max_tokens` limit as specified in `agen.yaml` & `qgen.yaml`. While other parameters can be changed as per your convenience.
# 8.  $N$ should be passed as an argument to `question_agent.py`. We'll test for $N=1$. `--num_questions` is the argument.
# 9.  Ensure **$40\%$** of the questions you generate gets filtered into `questions.json`.
# 
# 
# ### ALLOWED
# <!-- ✅  -->
# 1.  Participants are encouraged to modify the code scripts (for any sort of training, data construction, inference, such that the above constraints are not overruled).
# 2.  If you want to add `WANDB_API_KEY` for `wandb` logging do it in add `WANDB_API_KEY=xxxxxxx` before `python -m <script>.py` command. E.g., `!WANDB_API_KEY=xxxxxxx python -m agents.question_agent \`
# 

# %% [markdown]
# ## Token & Time Limit:
# <!-- ⏱  -->
# *   Maximum length (e.g., `max_token`) limit for your model response should be within following tokens.
#     *   For question-agent (Q-Agent) it is $100$ tokens cumulatively for the content corresponding to [`topic`, `question`, `choices`, and `answer`]. This excludes token count for double quotes as well as string length for topic, question, choices, and answer string it
#     *   And the rest is for explanation i.e., $1024-100 = 924$. But within time limit
# *   `ckpt` is the folder 📂 which you will place under `AAIPL_XXX_XXX_XXX_XXX`. While `checkpoints` folder 📂 inside `tutorial/` is meant for tutorial.
# *   Each question should be generated under `10 secs`. Overall time for 100 questions should be no more than `1000 secs` ~ `17 mins`.
# *   Each answer should be generated under `6 secs`. Overall time for 100 answers should be no more than `600 secs` ~ `10 mins`.
# *   *Note: We will only consider those questions' and answers' JSON file that remain under the time limit.*

# %% [markdown]
# ### Directory & Files overview
# <!-- 📂  -->
# ```
# .
# ├── agents
# │   ├── question_model.py
# │   ├── question_agent.py
# │   ├── answer_model.py
# │   └── answer_agent.py
# ├── tutorial # guide on how to SFT and GRPO
# │   ├── checkpoints #
# │   │     ├── sft # save sft checkpoints here while in tutorial
# │   │     ├── grpo # save grpo checkpoints here while in tutorial
# │   │     └── demo
# │   │         ├── sft # pre-trained sft (LoRA) ckpt
# │   │         └── grpo # same as above but for GRPO
# │   ├── tutorial.ipynb # guide on how to SFT and GRPO
# │   ├── trainer.py # sample training for Gemma with LoRA (SFT) and GRPO
# │   ├── answer_model2.py # inference script for the same. Copy it to agents/. for using this as question generator
# │   ├── formatted_questions_array.json # Sample question data for doing SFT and GRPO
# │   └── test_questions_array.json # Sample test question to evaluate the SFTed or GRPOed model
# ├── assets
# │   ├── topics_example.json # example questions w.r.t each topic
# │   ├── topics.json # Topics on which we require to generate questions
# │   ├── sample_question.json # File specifying expected format of questions generated
# │   ├── sample_answer.json # Expected format of answers generated
# │   └── AMDAAIPL.png # Teaser image for the AAIPL
# ├── utils
# │   └── build_prompt.py # prompt-tuning scripts
# ├── README.ipynb
# ├── outputs # That will consists of outputs from question_agent.py and answer_agent.py
# ├── ckpt # That will consists of checkpoints for question_agent.py and answer_agent.py if any training is done.
# ├── qgen.yaml # Generation specific parameters for Q-agent
# ├── agen.yaml # Generation specific parameters for A-agent
# └── default_requirements.txt # Packages required
# ```
#    

# %% [markdown]
# ### Env Setup
# <!-- 🚀 -->

# %%
# import basic packages
import json
from typing import Dict, Any, List

# %% [markdown]
# ### Let the GAME begin!!!
# <!-- 🎮  -->
# #### Q-Agent
# <!-- 🤔 -->
# <u>NOTE</u>: You are encouraged to invoke your own custom code into `question_model.py` and `question_agent.py` at `agents/`, to control its operation, respectively.

# %% [markdown]
# __Topics:__
# 1.  `Logical Reasoning`: Truth-teller and Liar Problems
# 2.  `Puzzles`: Seating Arrangements (Linear, Circular)
# 3.  `Blood Relations and Family Tree`: Puzzles involving generations and family tree logic
# 
# *To know what all topics are available, visit: **[topics.json](assets/topics.json)***

# %%
# Run the following code to generate questions.
# For demo purpose, we have used the base Qwen3-4B model for Q-Agent. Participants are expected to improve upon this
!python -m agents.question_agent \
    --output_file "outputs/questions.json" \
    --num_questions 20 \
    --verbose

# %% [markdown]
# #### Basic format-checks for questions from Q-agent
# 1. Here we filter questions into `questions.json` for usage of answer agent.
# 2. Further, the filtered questions will pass through an **`Oracle`** (a part of JUDGING system, hence closed and not demonstrated here) that checks *validity* of question, choices, and answer from Q-agent. It also provides the actual correct answer to the question.
# 3. BYOC (Bring Your Own Code): Well, again we emphasize to have your own innovations & code. Also the places with following tag/block or **similar**, expect some real improvements.
#     ```python
#     # TODO: IMPROVE THE FOLLOWING
#     <code>
#     ```
# 4. <span style="color : red">**Ensure**</span> on an average: $50\% \times \text{num\_questions} > N$ questions are filtered out.
# 5. The following filter is added into the `question_agent.py`. *<span style="color : red">Note that</span> we generate two version of questions, one is the usual, unfiltered one `questions.json` and the other is `filtered_questions.json` after passing through the below filter. <span style="color : green">We'll use this `filtered_questions.json` for conducting matches i.e., this file will be sent to opponent's answer agent. But do keep both in `outputs/` folder.</span>*
# 

# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/jupyter-tutorial/hf_models/Qwen3-4B", padding_side='left')

def count_tokens_q(text: str) -> int:
    """Count the number of tokens using Qwen3-4B tokenizer"""
    return len(tokenizer.encode(text, add_special_tokens=False))

def filter_questions(questions: List[str|Dict[str, str|Any]]) -> List[Dict[str, str|Any]]:
    def basic_checks(q2: Dict[str, str])->bool:
        # check required keys
        required_keys = ['topic', 'question', 'choices', 'answer']
        if all((key in q2) for key in required_keys):
            # check choices format
            checks = all(isinstance(choice, str) and len(choice) > 2 and choice[0].upper() in 'ABCD' for choice in q2['choices'])
            if isinstance(q2['choices'], list) and len(q2['choices']) == 4 and checks:
                # check answer format
                # Check token length
                check_len = sum(count_tokens_q(q2[k]) for k in ['question', 'answer'])
                check_len += sum(count_tokens_q(choice) for choice in q2['choices']) - 15
                if check_len < 130:
                    if check_len + count_tokens_q(q2.get('explanation', 'None')) <= 1024:
                        # Extra Checks: (PLUS checks) len(q2['answer']) == 1 and q2['answer'].upper() in 'ABCD':
                        if isinstance(q2['answer'], str):
                            return True
        return False
    correct_format_question = []
    for i, q in enumerate(questions):
        if isinstance(q, dict):
            if basic_checks(q):
                correct_format_question.append(q)
        elif isinstance(q, str):
            try:
                q1 = json.loads(q)
                if basic_checks(q1):
                    correct_format_question.append(q1)
            except json.JSONDecodeError:
                # If JSON decoding fails, skip this answer
                print(f"Skipping invalid JSON at index {i}: {q}")
                continue
        else:
            continue
    if len(correct_format_question) >= 0.5 * len(questions):
        return correct_format_question
    return list()

# %%
with open("outputs/questions.json", "r") as f:
    questions = json.load(f)

filtered_questions = filter_questions(questions)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Further filtering will happen with our Oracle (not shown here) which also have its own answer for the question.
# If Q-agent answer to its own question is wrong, then that question will not be considered.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

with open("outputs/filtered_questions.json", "w") as f:
    json.dump(filtered_questions, f, indent=4)

# %%
len(filtered_questions)

# %% [markdown]
# #### A-agent
# <!-- 🤖  -->
# <u>NOTE</u>: Here in the `answer_agent.py` you can integrate your custom model -- ***E.g., SFTed or GRPOed model [answer_model2.py](./tutorial/answer_model2.py)**. But first do SFT / GRPO -> load the checkpoint with correct path in [answer_model2.py](./tutorial/answer_model2.py) and then integrate it into `answer_agent.py`.*

# %%
# Same instructions apply for the answer agent.
# For demo purpose, we have used the base Qwen3-4B model for A-agent. Participants are expected to improve upon this.
!python -m agents.answer_agent \
    --input_file "outputs/filtered_questions.json" \
    --output_file "outputs/answers.json" \
    --verbose

# %% [markdown]
# #### Basic format-checks for answers from A-agent.
# 
# 1. Checks for expected `JSON` format as suggested in [instructions](#format-overview).
# 2. Same as Q-Agents, improvements are required here too.
# 3. Answers not having above format will not be considered and thus no point will be awarded.
# 4. The following filter is added into the `answer_agent.py`. Similarly here too, two versions are saved, `answers.json` and `filtered_answers.json`. The latter is used for evaluation.

# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/jupyter-tutorial/hf_models/Qwen3-4B", padding_side='left')

def count_tokens_a(text: str) -> int:
    """Count the number of tokens in the text using the agent's tokenizer"""
    return len(tokenizer.encode(text, add_special_tokens=False))

def filter_answers(ans: List[str|Dict[str, str]]) -> List[Dict[str, str]]:
    r"""Filter answers to ensure they are in the correct format"""
    def basic_checks(a1: Dict[str, str])->bool:
        # check required keys
        required_keys = ['answer']
        if all((key in a1) and isinstance(a1[key], str) for key in required_keys):
            if len(a1['answer']) == 1 and (a1['answer'] not in 'ABCDabcd'):
                    return False
            check_len = count_tokens_a(a1['answer'])
            if check_len < 50:
                check_len += count_tokens_a(a1.get('reasoning', 'None'))
                if check_len < 512:
                    # check answer format - EXTRA checks
                    # if len(a1['answer']) == 1 and a1['answer'].upper() in 'ABCD':
                    return True
        return False

    filtered_answers = []
    for i, a in enumerate(ans):
        if isinstance(a, dict):
            if basic_checks(a):
                filtered_answers.append(a)
            else:
                filtered_answers.append(None)
        elif isinstance(a, str):
            # Basic checks: at least with correct JSON format
            try:
                a1 = json.loads(a)
                if basic_checks(a1):
                    filtered_answers.append(a1)
                else:
                    filtered_answers.append(None)
            except json.JSONDecodeError:
                # If JSON decoding fails, skip this answer
                print(f"Skipping invalid JSON at index {i}: {a}")
                filtered_answers.append(None)
                continue
        else:
            # If the answer is neither a dict nor a str, skip it
            print(f"Skipping unsupported type at index {i}: {type(a)}")
            filtered_answers.append(None)
    return filtered_answers

# %%
with open("outputs/answers.json", "r") as f:
    answers = json.load(f)
filtered_answers = filter_answers(answers)

# %%
len(answers)

# %%
len(filtered_answers)

# %% [markdown]
# #### Evaluation
# <!-- 🏅  -->

# %% [markdown]
# ##### Scoring Criteria
# 
# <!-- 📊  -->
# 
# Simply, we assign scores based on, out of $N$ questions from Q-agent, how many an A-agent can answer and vice-versa. *No negative marking for wrong answers.*
# 
# $$\text{A-agent Score} = \dfrac{\#\ \text{of questions correctly answered with expected format}}{N}\times 100$$
# $$\text{Q-agent Score} = \dfrac{\#\ \text{of questions incorrectly answered by A-agent}}{N}\times 100$$

# %%
filtered_questions

# %%
for a in filtered_answers:
    if a is not None: print(a)

# %% [markdown]
# **An Example demonstrating how Q-agent matches up with A-agent**

# %%
# calculate scores...
N = len(filtered_questions)
assert N == len(filtered_answers), "Number of questions and answers must match."
num_correct_answers = len([1 for q,a in zip(filtered_questions, filtered_answers) if a is not None and q['answer'] == a['answer']])

# Here the answer may be correct, but since q['answer'] is not an option letter is not there, we face problems
# Below shown is one way of simple string parsing
num_correct_answers = len([1 for q,a in zip(filtered_questions, filtered_answers) if a is not None and q['answer'][0] == a['answer']])

a_score = num_correct_answers*100/(N+1e-9)
q_score = (N-num_correct_answers)*100/(N+1e-9)
# Announce the scores
print(f"Number of questions: {N}")
print(f"Number of correct answers: {num_correct_answers}")
print("Scores:")
print(f"Team B: A-agent score: {a_score:.2f}")
print(f"Team A: Q-agent score: {q_score:.2f}")
print(f"Innings 1 winner: {'Team A' if q_score > a_score else 'Team B' if q_score < a_score else 'Draw'}")
# DRAW case is not HANDLED now


