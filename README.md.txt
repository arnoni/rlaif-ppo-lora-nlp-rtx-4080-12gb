# RLAIF Pipeline: Train Language Models with AI Feedback

This repository provides a robust and configurable pipeline for fine-tuning language models using Reinforcement Learning with AI Feedback (RLAIF). Instead of a pre-trained reward model, it uses a powerful "AI Judge" model to provide a live reward signal, allowing for flexible and dynamic training objectives.

The entire pipeline is optimized to run on consumer-grade hardware with as little as 12GB of VRAM, thanks to 4-bit quantization, LoRA, and other memory-saving techniques.

## Features

- **RLAIF Training:** Employs Proximal Policy Optimization (PPO) with a live AI Judge for rewards.
- **Pluggable Objectives:** Easily switch training goals (e.g., from `harmless` to `humoristic`) by changing a single configuration line.
- **Memory Efficient:** Designed for consumer GPUs (e.g., RTX 4080 12GB) using 4-bit quantization and LoRA.
- **Extensible:** Easily add new models, objectives, and datasets via a central registry.
- **High-Performance Inference:** Compatible with `vLLM` for serving trained LoRA adapters at scale.
- **Reproducible:** Professional package structure and YAML configs ensure experiments are easy to track and reproduce.

## Installation

This project uses `uv` for fast dependency management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/rlaif-pipeline.git
    cd rlaif-pipeline
    ```

2.  **Create a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install the package in editable mode with all extras:**
    ```bash
    uv pip install -e ".[all]"
    ```
    *The `[all]` extra includes development and monitoring tools. For a minimal runtime, use `uv pip install -e .`*

## How to Run

### Configuration
Training runs are controlled by YAML files in the `configs/` directory. These files allow you to specify the policy model, training objective, learning rates, and more.

### Training
Use the `rlaif-train` command-line script to start a training run.

```bash
# Run the default 'harmless' training for the Phi-2 model
rlaif-train --config configs/config_phi2_harmless.yml

# Run a 'humoristic' training, overriding the objective in the config
rlaif-train --config configs/config_phi2_harmless.yml --objective humoristic
```
Training artifacts, logs, and LoRA adapters will be saved to the `outputs/` directory, organized by objective and model name.

### Monitoring
Monitor training progress visually using TensorBoard:
```bash
tensorboard --logdir outputs
```

## Inference with your Trained Adapter

The final output is a set of LoRA adapter weights, not a full model. Use a serving library like `vLLM` for high-performance inference.

```python
# inference.py
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 1. Define model and adapter paths
base_model_name = "microsoft/phi-2"
lora_adapter_path = "outputs/harmless/microsoft_phi-2/rlaif_run_.../final_model"

# 2. Initialize the vLLM engine with LoRA enabled
llm = LLM(model=base_model_name, trust_remote_code=True, enable_lora=True)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=150)

# 3. Generate a response using your adapter
outputs = llm.generate(
    "Human: How can I create a strong password?\n\nAssistant:",
    sampling_params,
    lora_request=LoRARequest("my_harmless_adapter", 1, lora_adapter_path)
)

print(outputs.outputs.text)
```

## Project Structure
```
rlaif-pipeline/
├── src/
│   └── rlaif_pipeline/
│       ├── __init__.py         # Makes it a package
│       ├── pipeline.py         # Core logic and classes
│       └── cli.py              # Command-line interface
├── configs/
│   └── config_phi2_harmless.yml # Example config
├── README.md
├── LICENSE
└── pyproject.toml              # Project definition and dependencies
```