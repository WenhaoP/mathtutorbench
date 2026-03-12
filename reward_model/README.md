# Reward Model Reproduction

This folder contains the reward-model artifacts used in our reproduction work.

## Files

- `compute_scaffolding_score.py`: released scoring script for pedagogical reward-model evaluation.
- `train_reward_model.py`: custom pairwise reward-model training script used for full-finetuning experiments.
- `train_reward_model_qlora.py`: custom QLoRA reward-model training script used for the 1.5B reproduction run.
- `TRAINING_RUNS.md`: recorded runs, hyperparameters, outcomes, and final metrics.

## Dataset

Both training scripts use the released dataset:

- `dmacjam/pedagogical-rewardmodel-data`

## Example commands

### 0.5B full finetuning

```bash
python reward_model/train_reward_model.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_path dmacjam/pedagogical-rewardmodel-data \
  --train_split train \
  --eval_split test \
  --output_dir /path/to/output \
  --bf16 \
  --gradient_checkpointing \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --max_length 1024
```

### 1.5B QLoRA

```bash
python reward_model/train_reward_model_qlora.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_path dmacjam/pedagogical-rewardmodel-data \
  --train_split train \
  --eval_split test \
  --output_dir /path/to/output \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --max_length 1024
```

## Notes

- The upstream repository releases the scorer and pretrained reward model, but not the original reward-model training pipeline.
- `train_reward_model.py` and `train_reward_model_qlora.py` are reproduction-oriented implementations.
- In our experiments, 0.5B full finetuning and 1.5B QLoRA completed successfully, while 1.5B full-parameter finetuning ran out of memory.
