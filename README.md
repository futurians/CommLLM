# Large Language Model Enhanced Multi-Agent Systems for 6G Communications

**This is a modified version of the original framework**
In order to work with it, you need to perform the steps described below.

## 1. Get the code and set up a virtual environment

Clone the repo: `git clone git@github.com:futurians/CommLLM.git`
Move to the cloned repo: `cd CommLLM`
Create a virtualenv: `python3 -m venv venv`
Activate the virtualenv: `source venv/bin/activate`
Install all required packages: `pip install -r requirements.txt`

## 2. Get llama-cpp

There are many ways to install – instructions are here: [https://github.com/ggml-org/llama.cpp/tree/master?tab=readme-ov-file#quick-start](https://github.com/ggml-org/llama.cpp/tree/master?tab=readme-ov-file#quick-start)
(Compiling from source has been working always, instructions for that are here: [https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md))

## 3. Run a llama-server with llama-cpp

This repo has been tested with two local models: Gemma 3 1B and Qwen 3 1.7B. The commands to run these models are as follows:

**Gemma 3 1B**

`llama-server -hf unsloth/gemma-3-1b-it-GGUF:Q4_K_M --chat-template gemma -ngl 99 --ctx-size 0 --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0`

**Qwen 3 1.7B**

`llama-server -hf unsloth/Qwen3-1.7B-GGUF --jinja --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 -c 40960 -n 32768 --no-context-shift -ngl 99`

Further information available in [local_models.md](./local_models.md)

## 4. Run the LLM-enhanced multi-agent system




## Authors
### Feibo Jiang, Yubo Peng, Li Dong, Kezhi Wang, Kun Yang, Cunhua Pan, Dusit Niyato, and Octavia A. Dobre.
## Paper
### https://arxiv.org/abs/2312.07850
## Code
### https://github.com/jiangfeibo/CommLLM.git
## Abstract
The rapid development of the large language model (LLM) presents huge opportunities for 6G communications — for example, network optimization and management — by allowing users to input task requirements to LLMs with natural language. However, directly applying native LLMs in 6G encounters various challenges, such as a lack of communication data and knowledge, and limited logical reasoning, evaluation, and refinement abilities. Integrating LLMs with the capabilities of retrieval, planning, memory, evaluation, and reflection in agents can greatly enhance the potential of LLMs for 6G communications. To this end, we propose CommLLM, a multi-agent system with customized communication knowledge and tools for solving communication-related tasks using natural language. This system consists of three components: multi-agent data retrieval (MDR), which employs the condensate and inference agents to refine and summarize communication knowledge from the knowledge base, expanding the knowledge boundaries of LLMs in 6G communications; multi-agent collaborative planning (MCP), which utilizes multiple planning agents to generate feasible solutions for the communication-related task from different perspectives based on the retrieved knowledge; and multi-agent evaluation and reflection (MER), which utilizes the evaluation agent to assess the solutions, and applies the reflection agent and refinement agent to provide improvement suggestions for current solutions. Finally, we validate the effectiveness of the proposed multiagent system by designing a semantic communication system as a case study of 6G communications.
![img](SC.png)

## The function of each file
- [LLMMAS.ipynb](LLMMAS.ipynb): The implementation of LLM-enhanced multi-agent systems for the generation of semantic communication systems.

- [Test_generated_SC.py](Test_generated_SC.py): Test the generated SC model based on cosine similarity.

- [1.pdf](1.pdf): Reference paper.

- [code.txt](code.txt): Reference code.

- [movie_lines.txt](movie_lines.txt): Training and test data for the generated SC model.

## Citation   
```
@ARTICLE{10638533,
  author={Jiang, Feibo and Peng, Yubo and Dong, Li and Wang, Kezhi and Yang, Kun and Pan, Cunhua and Niyato, Dusit and Dobre, Octavia A.},
  journal={IEEE Wireless Communications}, 
  title={Large Language Model Enhanced Multi-Agent Systems for 6G Communications}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/MWC.016.2300600}}
```

