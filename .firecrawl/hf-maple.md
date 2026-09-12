[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)Hugging Face](https://huggingface.co/)

- [Models](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Spaces](https://huggingface.co/spaces)
- [Buckets new](https://huggingface.co/storage)
- [Docs](https://huggingface.co/docs)
- [Enterprise](https://huggingface.co/enterprise)
- [Pricing](https://huggingface.co/pricing)
- - Website

    - [Tasks](https://huggingface.co/tasks)
    - [HuggingChat](https://huggingface.co/chat)
    - [Collections](https://huggingface.co/collections)
    - [Languages](https://huggingface.co/languages)
    - [Organizations](https://huggingface.co/organizations)
  - Community

    - [Blog](https://huggingface.co/blog)
    - [Posts](https://huggingface.co/posts)
    - [Daily Papers](https://huggingface.co/papers)
    - [Hardware](https://huggingface.co/hardware)
    - [Learn](https://huggingface.co/learn)
    - [Discord](https://huggingface.co/join/discord)
    - [Forum](https://discuss.huggingface.co/)
    - [GitHub](https://github.com/huggingface)
  - Solutions

    - [Team & Enterprise](https://huggingface.co/enterprise)
    - [Hugging Face PRO](https://huggingface.co/pro)
    - [Enterprise Support](https://huggingface.co/support)
    - [Inference Providers](https://huggingface.co/inference/models)
    - [Inference Endpoints](https://huggingface.co/inference-endpoints)
    - [Storage Buckets](https://huggingface.co/storage)

- * * *

- [Log In](https://huggingface.co/login)
- [Sign Up](https://huggingface.co/join)

# [![](https://cdn-avatars.huggingface.co/v1/production/uploads/67ca738587a625b6a9affd0f/lMDrDwMactisn7AGYDKbm.png)](https://huggingface.co/deepgrove)  [deepgrove](https://huggingface.co/deepgrove)  /      [maple-preview](https://huggingface.co/deepgrove/maple-preview)    like382           Follow ![](https://cdn-avatars.huggingface.co/v1/production/uploads/67ca738587a625b6a9affd0f/lMDrDwMactisn7AGYDKbm.png)deepgrove287

[Text Generation](https://huggingface.co/models?pipeline_tag=text-generation) [Transformers](https://huggingface.co/models?library=transformers) [Safetensors](https://huggingface.co/models?library=safetensors) [English](https://huggingface.co/models?language=en) [causal-lm](https://huggingface.co/models?other=causal-lm) [mixture-of-experts](https://huggingface.co/models?other=mixture-of-experts) [reasoning](https://huggingface.co/models?other=reasoning) [ternary](https://huggingface.co/models?other=ternary) [custom-code](https://huggingface.co/models?other=custom-code) [conversational](https://huggingface.co/models?other=conversational) [custom\_code](https://huggingface.co/models?other=custom_code)

License:mit

[Model card](https://huggingface.co/deepgrove/maple-preview) [FilesFiles and versions\\
xet](https://huggingface.co/deepgrove/maple-preview/tree/main) [Community\\
9](https://huggingface.co/deepgrove/maple-preview/discussions)

Deploy

Copy to bucket new

Use this model

### Instructions to use deepgrove/maple-preview with libraries, inference providers, notebooks, and local apps. Follow these links to get started.

- Libraries
- [Transformers](https://huggingface.co/deepgrove/maple-preview?library=transformers)
How to use deepgrove/maple-preview with Transformers:


```
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="deepgrove/maple-preview", trust_remote_code=True)
messages = [\
      {"role": "user", "content": "Who are you?"},\
]
pipe(messages)
```



```
# Load model directly
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("deepgrove/maple-preview", trust_remote_code=True, device_map="auto")
```

- Notebooks
- [Google Colab](https://huggingface.co/deepgrove/maple-preview/colab)
- [Kaggle](https://huggingface.co/deepgrove/maple-preview/kaggle)
- Local Apps [Settings](https://huggingface.co/settings/local-apps "Set up your favorite local applications")
- [vLLM](https://huggingface.co/deepgrove/maple-preview?local-app=vllm)
How to use deepgrove/maple-preview with vLLM:


##### Install from pip and serve model



```
# Install vLLM from pip:
pip install vllm
# Start the vLLM server:
vllm serve "deepgrove/maple-preview"
# Call the server using curl (OpenAI-compatible API):
curl -X POST "http://localhost:8000/v1/chat/completions" \
  	-H "Content-Type: application/json" \
  	--data '{
  		"model": "deepgrove/maple-preview",
  		"messages": [\
  			{\
  				"role": "user",\
  				"content": "What is the capital of France?"\
  			}\
  		]
  	}'
```



##### Use Docker



```
docker model run hf.co/deepgrove/maple-preview
```

- [SGLang](https://huggingface.co/deepgrove/maple-preview?local-app=sglang)
How to use deepgrove/maple-preview with SGLang:


##### Install from pip and serve model



```
# Install SGLang from pip:
pip install sglang
# Start the SGLang server:
python3 -m sglang.launch_server \
      --model-path "deepgrove/maple-preview" \
      --host 0.0.0.0 \
      --port 30000
# Call the server using curl (OpenAI-compatible API):
curl -X POST "http://localhost:30000/v1/chat/completions" \
  	-H "Content-Type: application/json" \
  	--data '{
  		"model": "deepgrove/maple-preview",
  		"messages": [\
  			{\
  				"role": "user",\
  				"content": "What is the capital of France?"\
  			}\
  		]
  	}'
```



##### Use Docker images



```
docker run --gpus all \
      --shm-size 32g \
      -p 30000:30000 \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      --env "HF_TOKEN=<secret>" \
      --ipc=host \
      lmsysorg/sglang:latest \
      python3 -m sglang.launch_server \
          --model-path "deepgrove/maple-preview" \
          --host 0.0.0.0 \
          --port 30000
# Call the server using curl (OpenAI-compatible API):
curl -X POST "http://localhost:30000/v1/chat/completions" \
  	-H "Content-Type: application/json" \
  	--data '{
  		"model": "deepgrove/maple-preview",
  		"messages": [\
  			{\
  				"role": "user",\
  				"content": "What is the capital of France?"\
  			}\
  		]
  	}'
```

- [Docker Model Runner](https://huggingface.co/deepgrove/maple-preview?local-app=docker-model-runner)
How to use deepgrove/maple-preview with Docker Model Runner:


```
docker model run hf.co/deepgrove/maple-preview
```

- [Browse\\
Quantizations](https://huggingface.co/models?other=base_model:quantized:deepgrove/maple-preview) to use this model in  llama.cpp,  Ollama,  LM Studio, or any compatible app.


- [Maple-Preview](https://huggingface.co/deepgrove/maple-preview#maple-preview "Maple-Preview")
  - [Architecture](https://huggingface.co/deepgrove/maple-preview#architecture "Architecture")
  - [Evaluation](https://huggingface.co/deepgrove/maple-preview#evaluation "Evaluation")
  - [Limitations](https://huggingface.co/deepgrove/maple-preview#limitations "Limitations")
  - [License](https://huggingface.co/deepgrove/maple-preview#license "License")

# Maple-Preview

**DeepGrove · 2026**

Today we introduce Maple-Preview, an open-source 20B-A1B ternary-weight reasoning LLM. Maple-Preview has SOTA reasoning for its weight class and is even competitive with larger models. It solves IMO-level problems and runs at 200+ tokens/sec on a Mac mini M4, 5–16× faster than efficient models like Gemma 4, Qwen3.5, and gpt-oss.

- 20B-A1B Model
- 218 tok/s M4 Mac mini
- 5.31 GB Checkpoint
- 131,072 Token context

[![Maple-Preview speed and performance frontier](https://huggingface.co/deepgrove/maple-preview/resolve/main/assets/01-speed-frontier.png)](https://huggingface.co/deepgrove/maple-preview/blob/main/assets/01-speed-frontier.png)

> The included Transformers implementation depends on Triton and FlashAttention
> and is intended for a compatible CUDA environment. The reported Apple Silicon
> result uses a separate on-device runtime.

## Architecture

Maple-Preview is a 20B-A1B reasoning model designed from the start for efficient on-device inference. It utilizes a 24-layer, 256-expert (8 active) configuration with 3:1 SWA-512:GA attention.

## Evaluation

On benchmarks, Maple-Preview sets a new point on the Pareto frontier for both memory-to-performance and speed-to-performance, demonstrating its strong reasoning capabilities. However, we note that this preview is focused primarily on raw reasoning and, as such, may underperform on agentic benchmarks. We intend to continue improving general performance through extended training before Maple's full release.

[![Benchmark score comparison](https://huggingface.co/deepgrove/maple-preview/resolve/main/assets/05-benchmark-scores-table.png)](https://huggingface.co/deepgrove/maple-preview/blob/main/assets/05-benchmark-scores-table.png)

Capability comparison using the dense output head across LCBv6, AIME 2026, HMMT 2026, and GPQA-D.

## Limitations

This preview received minimal post-training for agentic tasks and only
small-scale general reinforcement learning.

## License

Maple-Preview is released under the [MIT License](https://huggingface.co/deepgrove/maple-preview/tree/main/LICENSE).

Downloads last month7,278

Safetensors

Model size

20B params

Tensor type

BF16

·

Chat template

Files info

Inference Providers [NEW](https://huggingface.co/docs/inference-providers)

[Text Generation](https://huggingface.co/tasks/text-generation "Learn more about text-generation")

This model isn't deployed by any Inference Provider. [🙋3Ask for provider support](https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/11633)

## Model tree for deepgrove/maple-preview

Finetunes

[3 models](https://huggingface.co/models?other=base_model:finetune:deepgrove/maple-preview)

Quantizations

[Use with llama.cpp](https://huggingface.co/models?apps=llama.cpp&other=base_model:quantized:deepgrove/maple-preview "Use with llama.cpp")[Use with LM Studio](https://huggingface.co/models?apps=lmstudio&other=base_model:quantized:deepgrove/maple-preview "Use with LM Studio")[Use with Jan](https://huggingface.co/models?apps=jan&other=base_model:quantized:deepgrove/maple-preview "Use with Jan")[Use with Ollama](https://huggingface.co/models?apps=ollama&other=base_model:quantized:deepgrove/maple-preview "Use with Ollama")

[13 models](https://huggingface.co/models?other=base_model:quantized:deepgrove/maple-preview)

## Spaces using deepgrove/maple-preview2

[🍁\\
\\
ProCreations/maple-webgpu](https://huggingface.co/spaces/ProCreations/maple-webgpu) [🍁\\
\\
ajsbsd/maple-preview](https://huggingface.co/spaces/ajsbsd/maple-preview)

## Collection including deepgrove/maple-preview

[**Maple-Preview**\\
\\
Collection\\
\\
A 20B-A1B reasoning model.•3 items•Updated 26 days ago• 8](https://huggingface.co/collections/deepgrove/maple-preview)

System theme

Company

[TOS](https://huggingface.co/terms-of-service) [Privacy](https://huggingface.co/privacy) [About](https://huggingface.co/huggingface) [Careers](https://apply.workable.com/huggingface/)  [Hugging Face](https://huggingface.co/)

Website

[Models](https://huggingface.co/models) [Datasets](https://huggingface.co/datasets) [Spaces](https://huggingface.co/spaces) [Pricing](https://huggingface.co/pricing) [Docs](https://huggingface.co/docs)

StripeM-Inner

Inference providers allow you to run inference using different serverless providers.