# Reward Model Training Runs

This document records the reward-model training runs used for reproduction work outside the upstream repository.

Notes:
- The upstream MathTutorBench repository releases reward-model scoring code and a pretrained reward model, but it does not release the original reward-model training pipeline.
- The runs below use a custom pairwise reward-model training script built during reproduction.
- Hyperparameters below reflect the actual Colab commands used during experimentation, including failed runs caused by GPU memory limits.

## Shared Setup

- Objective: pairwise reward modeling with logistic loss on preferred vs. rejected teacher responses
- Dataset: `dmacjam/pedagogical-rewardmodel-data`
- Train split: `train`
- Eval split: `test`
- Logging backend: Weights & Biases (`mathtutorbench-rm`)
- Output root on Colab: `/content/drive/MyDrive/mathtutorbench_repro/rm/`
- Training implementation: custom `reward_model/train_reward_model.py`

## Run 1: `qwen2.5-1.5b-rm`

Status: failed with CUDA OOM during backward pass

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Output dir: `/content/drive/MyDrive/mathtutorbench_repro/rm/qwen2.5-1.5b-rm`
- Run name: `qwen2.5-1.5b-rm`
- Report to: `wandb`
- Precision: `bf16`
- Gradient checkpointing: disabled
- Per-device train batch size: `1`
- Per-device eval batch size: `1`
- Gradient accumulation steps: `16`
- Effective train batch size: `16`
- Learning rate: `1e-5`
- Epochs: `1`
- Max length: `2048`
- Logging steps: `10`
- Eval steps: `100`
- Save steps: `100`
- Save total limit: `2`
- Optimizer: Hugging Face Trainer default `AdamW`

## Run 2: `qwen2.5-1.5b-rm-l1024`

Status: failed with CUDA OOM during optimizer step

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Output dir: `/content/drive/MyDrive/mathtutorbench_repro/rm/qwen2.5-1.5b-rm`
- Run name: `qwen2.5-1.5b-rm-l1024`
- Report to: `wandb`
- Precision: `bf16`
- Gradient checkpointing: enabled
- Per-device train batch size: `1`
- Per-device eval batch size: `1`
- Gradient accumulation steps: `16`
- Effective train batch size: `16`
- Learning rate: `1e-5`
- Epochs: `1`
- Max length: `1024`
- Logging steps: `10`
- Eval steps: `200`
- Save steps: `200`
- Save total limit: `2`
- Optimizer: Hugging Face Trainer default `AdamW`

## Run 3: `qwen2.5-0.5b-rm-l1024`

Status: completed successfully

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Output dir: `/content/drive/MyDrive/mathtutorbench_repro/rm/qwen2.5-0.5b-rm`
- Run name: `qwen2.5-0.5b-rm-l1024`
- Report to: `wandb`
- Precision: `bf16`
- Gradient checkpointing: enabled
- Per-device train batch size: `1`
- Per-device eval batch size: `1`
- Gradient accumulation steps: `16`
- Effective train batch size: `16`
- Learning rate: `1e-5`
- Epochs: `1`
- Max length: `1024`
- Logging steps: `10`
- Eval steps: `200`
- Save steps: `200`
- Save total limit: `2`
- Optimizer: Hugging Face Trainer default `AdamW`
- W&B project: `mathtutorbench-rm`
- Final eval preference accuracy: `0.8195`
- Final eval loss: `0.4626`
- Final train loss: `0.2204`

## Run 4: `qwen2.5-1.5b-rm-qlora`

Status: completed successfully with QLoRA

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Output dir: `/content/drive/MyDrive/mathtutorbench_repro/rm/qwen2.5-1.5b-rm-qlora`
- Run name: `qwen2.5-1.5b-rm-qlora`
- Report to: `wandb`
- Precision: `bf16`
- Quantization: `4-bit NF4`
- Adapter method: `QLoRA`
- Per-device train batch size: `1`
- Per-device eval batch size: `1`
- Gradient accumulation steps: `16`
- Effective train batch size: `16`
- Learning rate: `2e-4`
- Epochs: `1`
- Max length: `1024`
- Logging steps: `10`
- Eval steps: `200`
- Save steps: `200`
- Save total limit: `2`
- Optimizer: `paged_adamw_8bit`
- LoRA rank (`r`): `16`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Final eval preference accuracy: `0.8050`
- Final eval mean chosen score: `4.1582`

## Reproduction Caveat

These runs should be described as reproduction-oriented experiments rather than exact execution of the authors' original reward-model training code. The authors released the benchmark, scoring script, dataset reference, and pretrained reward model, but not the original RM training script or full hyperparameter configuration.
