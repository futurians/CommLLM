# Running LLMMAS with local models

The code in this repository has been modified to make it run with a local model. Testing has been performed with `llama.cpp` – specifically with the `llama-server`-component that comes with this inference backend. The server commands used to spin up a server (along with other details) that were used during the testing are given below.

Note that `llama-server` will handle downloading the model, if the provided commands are used.

## Gemma 3 1B

The model in GGUF-format was obtained from (https://huggingface.co/unsloth/gemma-3-1b-it-GGUF)[https://huggingface.co/unsloth/gemma-3-1b-it-GGUF]

The server command used was: `llama-server -hf unsloth/gemma-3-1b-it-GGUF:Q4_K_M --chat-template gemma -ngl 99 --ctx-size 0 --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0`

## Qwen 3 1.7B

The model in GGUF-format was obtained from (https://huggingface.co/unsloth/Qwen3-1.7B-GGUF)[https://huggingface.co/unsloth/Qwen3-1.7B-GGUF]

Model was run with server command `llama-server -hf unsloth/Qwen3-1.7B-GGUF --jinja --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 -c 40960 -n 32768 --no-context-shift -ngl 99`