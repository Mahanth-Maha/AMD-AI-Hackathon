# AMD-AI-Hackathon
AMD AI Hackathon project 


---

### Contributors :
* Yalla Mahanth 
* Md Kaif Alam 
* Pritam Kumar
---


## Setup 

1. Clone the repository:
```bash
git clone https://github.com/Mahanth-Maha/AMD-AI-Hackathon.git
cd AMD-AI-Hackathon
```

2. Create a virtual environment:
```bash 
conda create -n amd_ai python=3.10.18 -y 
conda activate amd_ai
```

2.a. make noteboook kernel:
```bash
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name amd_ai --display-name "amd_ai"
``` 

3. Install the required packages:
```bash
pip install -r requirements.txt
```


## Hackathon Overview

### Track 1 : AMD AI Premier League (AAIPL)

A head-to-head AI competition where teams of up to 3 developers build two intelligent language model based agents:

* ​Q-agent: Generates valid, challenging multiple choice questions belonging to a given domain.

* ​A-agent: Attempts to answer questions posed by the opposing team’s Q-agent.

* ​Goal: Create a question generator that can pose the most difficult yet correct questions while ensuring your answerer can accurately answer as many of them.

#### ​Format:

​Matches are played between pairs of teams where one team's Q-agent generates a set of questions to which A-agent of the opposing team responds and vice-versa.

​Winning teams advance to the next stage.

#### ​Resources Provided:

* ​1 MI300 GPU for 24 hours

* Sample code for prompt tuning, reinforcement learning, and fine-tuning

#### ​Deliverables:

​Final working solution

​PowerPoint summarizing techniques used



