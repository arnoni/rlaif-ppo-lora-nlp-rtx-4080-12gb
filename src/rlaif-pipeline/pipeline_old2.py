# src/rlaif_pipeline/pipeline.py

import argparse
import json
import logging
import os
import random
import re
import sqlite3
import sys
import time
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import numpy as np

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None
    PYNVML_AVAILABLE = False

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging as hf_logging,
    GenerationConfig,
)
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model, TaskType

# ==============================================================================
# CONFIGURATION
# ==============================================================================
@dataclass
class ModelConfig:
    policy_model_name: str = "microsoft/phi-2"
    judge_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    fallback_model_name: str = "microsoft/DialoGPT-small"
    quantization_bits: int = 4
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True

@dataclass
class LoRAConfig:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.1
    target_modules: List[str] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

@dataclass
class TrainingConfig:
    objective: str = "harmless"
    run_name: str = f"rlaif_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir: str = "outputs"
    total_steps: int = 1000
    batch_size: int = 2
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1.41e-5
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    eval_interval: int = 50
    log_interval: int = 10
    save_interval: int = 200
    max_vram_gb: float = 11.0
    
@dataclass
class DataConfig:
    dataset_name: str = "Anthropic/hh-rlhf"
    train_split: str = "train[:10%]"
    max_prompt_tokens: int = 256
    max_response_tokens: int = 384
    streaming: bool = True

@dataclass
class RLAIFConfig:
    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    data: DataConfig
    
    def save(self, path: str):
        with open(path, 'w') as f: yaml.dump(asdict(self), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f: data = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**data.get('model', {})),
            lora=LoRAConfig(**data.get('lora', {})),
            training=TrainingConfig(**data.get('training', {})),
            data=DataConfig(**data.get('data', {}))
        )

# ==============================================================================
# OBJECTIVE REGISTRY
# ==============================================================================
OBJECTIVE_REGISTRY = {
    "harmless": {
        "dataset_name": "Anthropic/hh-rlhf", "dataset_split": "train[:10%]", "dataset_prompt_field": "chosen",
        "prompt_template": """Rate the harmlessness of the AI response on a scale from 0 to 10. A harmless response is safe, ethical, and avoids toxic, dangerous, or offensive content.
User Prompt: {prompt}
AI Response: {response}
Provide ONLY a single numeric score from 0 to 10. Score:"""
    },
    "humoristic": {
        "dataset_name": "or-dvir/r-jokes", "dataset_split": "train", "dataset_prompt_field": "text",
        "prompt_template": """As a comedy critic, rate the humor of the AI response on a scale from 0 to 10. A high score means the response is witty, clever, and genuinely funny.
User Prompt: {prompt}
AI Response: {response}
Provide ONLY a single numeric score from 0 to 10. Score:"""
    },
    "philosophic": {
        "dataset_name": "laion/strategic_reasoning_and_philosophy", "dataset_split": "train[:20%]", "dataset_prompt_field": "text",
        "prompt_template": """As a philosophy professor, rate the depth and clarity of the reasoning in this response on a scale of 0-10.
User Prompt: {prompt}
AI Response: {response}
Provide ONLY a single numeric score from 0 to 10. Score:"""
    },
}

# ==============================================================================
# UTILITIES AND HELPER CLASSES
# (Logger, MemoryManager, RewardCache, etc.)
# ==============================================================================
logger = logging.getLogger(__name__)

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers(): logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] [%(name)s] %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    hf_logging.set_verbosity_error()
    return logger

class MemoryManager:
    # ... (Implementation from v4 is unchanged)
    def __init__(self, max_vram_gb: float = 11.0):
        self.max_vram_gb = max_vram_gb
        if PYNVML_AVAILABLE: pynvml.nvmlInit()
            
    def get_memory_info_gb(self) -> Dict[str, float]:
        if not PYNVML_AVAILABLE or not torch.cuda.is_available(): return {"total": 0, "used": 0, "free": 0}
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {"total": mem_info.total / 1e9, "used": mem_info.used / 1e9, "free": mem_info.free / 1e9}

    def shutdown(self):
        if PYNVML_AVAILABLE: pynvml.nvmlShutdown()

class EnhancedRewardCache:
    # ... (Implementation from v4 is unchanged)
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.RewardCache")
        self._init_db()
        
    def _init_db(self):
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS rewards (response_hash TEXT PRIMARY KEY, reward REAL, model_version TEXT)
            """)
    
    @staticmethod
    def _hash_text(text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get(self, response: str) -> Optional[float]:
        rh = self._hash_text(response)
        cur = self.conn.cursor()
        cur.execute("SELECT reward FROM rewards WHERE response_hash=?", (rh,))
        row = cur.fetchone()
        return row[0] if row else None
    
    def set(self, response: str, reward: float, model_version: str):
        with self.conn:
            rh = self._hash_text(response)
            self.conn.execute("INSERT OR REPLACE INTO rewards VALUES (?, ?, ?)", (rh, reward, model_version))
    
    def close(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.logger.info("Reward cache connection closed.")


class PPOPromptDataset(IterableDataset):
    def __init__(self, config: DataConfig, tokenizer: AutoTokenizer, objective: str):
        self.config = config
        self.tokenizer = tokenizer
        self.objective_details = OBJECTIVE_REGISTRY[objective]
        self.prompt_field = self.objective_details["dataset_prompt_field"]
        self._load_dataset()
    
    def _load_dataset(self):
        self.dataset = load_dataset(self.config.dataset_name, split=self.config.train_split, streaming=True)

    def __iter__(self):
        for raw_example in self.dataset:
            dialogue = raw_example.get(self.prompt_field, "")
            if not isinstance(dialogue, str): continue
            
            if self.objective_details["dataset_name"] == "Anthropic/hh-rlhf":
                prompt_str = dialogue.split("\n\nAssistant:")[0] + "\n\nAssistant:"
            else:
                prompt_str = f"Human: {dialogue}\n\nAssistant:"

            tokenized_prompt = self.tokenizer(prompt_str, truncation=True, max_length=self.config.max_prompt_tokens, return_tensors="pt")
            yield {"query": prompt_str, "input_ids": tokenized_prompt["input_ids"].squeeze(0)}

class EnhancedAIJudge:
    def __init__(self, config: ModelConfig, device: Any):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(f"{__name__}.AIJudge")
        self.tokenizer, self.model, self.cache = None, None, None
        self.model_version = f"{config.judge_model_name}-{config.quantization_bits}bit"
    
    def load(self, cache_path: str):
        self.cache = EnhancedRewardCache(cache_path)
        for model_name in [self.config.judge_model_name, self.config.fallback_model_name]:
            try:
                self.model, self.tokenizer = self._load_model(model_name)
                return
            except Exception as e:
                self.logger.warning(f"Failed to load {model_name}: {e}")
        raise RuntimeError("Failed to load any judge model")

    def _load_model(self, model_name: str):
        self.logger.info(f"Loading judge model: {model_name}")
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map=self.device, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def _create_scoring_prompt(self, prompt: str, response: str, objective: str) -> str:
        template = OBJECTIVE_REGISTRY[objective]["prompt_template"]
        return template.format(prompt=prompt, response=response)

    def _extract_score(self, text: str) -> float:
        match = re.search(r'^\s*(\d+\.?\d*)', text)
        if match: return float(match.group(1))
        return 5.0 # Default neutral score

    def score_responses(self, prompts: List[str], responses: List[str], objective: str) -> List[float]:
        scores = []
        for prompt, response in zip(prompts, responses):
            cached_score = self.cache.get(response)
            if cached_score is not None:
                scores.append(cached_score)
                continue
            
            gen_config = GenerationConfig(max_new_tokens=10, pad_token_id=self.tokenizer.pad_token_id)
            scoring_prompt = self._create_scoring_prompt(prompt, response, objective)
            inputs = self.tokenizer(scoring_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=gen_config)
            
            decoded = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            score = self._extract_score(decoded)
            normalized_score = max(0.0, min(1.0, score / 10.0))
            self.cache.set(response, normalized_score, self.model_version)
            scores.append(normalized_score)
        return scores
        
    def close_cache(self):
        if self.cache: self.cache.close()

def get_dynamic_lora_targets(model_name: str) -> List[str]:
    name = model_name.lower()
    if "llama" in name or "mistral" in name: return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "phi" in name: return ["q_proj", "v_proj", "k_proj", "dense", "fc1", "fc2"]
    if "gpt2" in name or "gpt-neo" in name: return ["c_attn", "c_proj"]
    return ["q_proj", "v_proj"]

def build_enhanced_policy_model(config: RLAIFConfig) -> Tuple[Any, AutoTokenizer]:
    logger.info(f"Building policy model: {config.model.policy_model_name}")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model.policy_model_name,
        quantization_config=quant_config, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="flash_attention_2" if config.model.use_flash_attention else "sdpa",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.policy_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    lora_config = LoraConfig(
        r=config.lora.r, lora_alpha=config.lora.alpha, lora_dropout=config.lora.dropout,
        target_modules=get_dynamic_lora_targets(config.model.policy_model_name),
        bias=config.lora.bias, task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    
    if config.model.gradient_checkpointing: model.gradient_checkpointing_enable()
    return model, tokenizer

# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================
def main(config: RLAIFConfig):
    # Setup
    objective = config.training.objective
    sanitized_model_name = config.model.policy_model_name.replace("/", "_")
    output_dir = Path(config.training.output_dir) / objective / sanitized_model_name / config.training.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    global logger
    logger = setup_logger(__name__, log_file=str(output_dir / "training.log"))
    
    logger.info("=" * 65)
    logger.info("STARTING RLAIF TRAINING PIPELINE V4".center(65))
    logger.info("=" * 65)
    logger.info(f"Objective: {objective.upper()}")
    logger.info(f"Policy Model: {config.model.policy_model_name}")
    logger.info(f"Full output path: {output_dir.resolve()}")

    config.data.dataset_name = OBJECTIVE_REGISTRY[objective]["dataset_name"]
    config.data.train_split = OBJECTIVE_REGISTRY[objective]["dataset_split"]
    config.save(str(output_dir / "config.yml"))

    mem_manager = MemoryManager(config.training.max_v_gb)
    ai_judge = None
    
    try:
        ppo_config = PPOConfig(
            steps=config.training.total_steps,
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            mini_batch_size=config.training.mini_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            log_with="tensorboard",
            project_kwargs={"logging_dir": str(output_dir / "tensorboard_logs")},
        )

        policy_model, tokenizer = build_enhanced_policy_model(config)
        dataset = PPOPromptDataset(config.data, tokenizer, objective)
        
        ppo_trainer = PPOTrainer(config=ppo_config, model=policy_model, ref_model=None, tokenizer=tokenizer, dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
        
        ai_judge = EnhancedAIJudge(config.model, device=ppo_trainer.accelerator.device)
        ai_judge.load(cache_path=str(output_dir / "rewards.db"))

        gen_kwargs = {"max_new_tokens": config.data.max_response_tokens, "do_sample": True, "top_k": 50, "top_p": 0.9, "temperature": 0.7, "pad_token_id": tokenizer.pad_token_id}

        logger.info("--- Starting PPO Training Loop ---")
        for step, batch in enumerate(ppo_trainer.dataloader):
            if step >= config.training.total_steps: break

            query_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **gen_kwargs)
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            batch["query"] = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
            
            rewards = ai_judge.score_responses(batch["query"], batch["response"], objective=objective)
            reward_tensors = [torch.tensor(r, device=ppo_trainer.accelerator.device) for r in rewards]
            
            stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
            
            if step > 0 and step % config.training.log_interval == 0:
                stats['env/reward_mean'] = torch.mean(torch.stack(reward_tensors)).cpu().item()
                ppo_trainer.log_stats(stats, batch, reward_tensors)
                mem_info = mem_manager.get_memory_info_gb()
                logger.info(f"Step {step}/{config.training.total_steps} | Reward: {stats['env/reward_mean']:.3f} | VRAM: {mem_info['used']:.2f}GB")

            if step > 0 and step % config.training.save_interval == 0:
                ppo_trainer.save_pretrained(str(output_dir / f"checkpoint_{step}"))

        final_model_dir = output_dir / "final_model"
        ppo_trainer.save_pretrained(str(final_model_dir))
        logger.info("="*65)
        logger.info("TRAINING COMPLETE".center(65))
        logger.info(f"Final LoRA adapter saved to: {final_model_dir.resolve()}")
        logger.info("="*65)

    except Exception as e:
        logger.error("An unrecoverable error occurred during training.", exc_info=True)
    finally:
        logger.info("Shutting down resources...")
        if ai_judge: ai_judge.close_cache()
        if mem_manager: mem_manager.shutdown()