deepgrove-ai/mlx-lm-deepgrove - GitHub
  URL: https://github.com/deepgrove-ai/mlx-lm-deepgrove
  # Maple on mlx-lm
## Setup
```
git clone git@github.com:deepgrove-ai/mlx-lm-deepgrove.git
cd mlx-lm-deepgrove
./setup.sh
source .venv/bin/activate
hf download deepgrove/maple-2bit-mlx --local-dir maple-2bit-mlx
```

## Run
```
python -m mlx_lm generate --model ./maple-2bit-mlx --trust-remote-code --flash-head \
  --prompt "Write a haiku about a grove." --temp 1.0 --top-p 0.95 --top-k 20

python -m mlx_lm chat --model ./maple-2bit-mlx --trust-remote-code --max-tokens -1 \
  --temp 1.0 --top-p 0.95
```
  Category: github

llama.cpp/README.md at main · deepgrove-ai/llama.cpp · GitHub
  URL: https://github.com/deepgrove-ai/llama.cpp/blob/main/README.md
  --jinja applies Maple's embedded chat template exactly, including its thinking prefix. Benchmark. M5 Pro, CPU-only, 16 threads, 512 prompt tokens, 128 ...
  Category: github
