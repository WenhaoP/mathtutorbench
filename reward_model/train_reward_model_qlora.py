import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Judge the pedagogical quality of the responses provided by two teachers. "
    "Focus on the quality of the scaffolding guidance, correctness, and "
    "actionability of the feedback through nudges, questions and hints. "
    "Do not give high scores for revealing the full answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a QLoRA pedagogical reward model.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--dataset_path", type=str, default="dmacjam/pedagogical-rewardmodel-data")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--validation_size", type=float, default=0.05)

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def load_preference_splits(args: argparse.Namespace) -> DatasetDict:
    dataset = load_dataset(args.dataset_path, args.dataset_name)
    if args.train_split not in dataset:
        raise ValueError(f"Missing train split '{args.train_split}' in dataset {args.dataset_path}.")

    train_dataset = dataset[args.train_split]
    if args.eval_split in dataset:
        eval_dataset = dataset[args.eval_split]
    elif args.validation_size > 0:
        split = train_dataset.train_test_split(test_size=args.validation_size, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        eval_dataset = train_dataset.select([])

    return DatasetDict({"train": train_dataset, "eval": eval_dataset})


def conversation_to_text(conversation: List[Dict[str, str]]) -> str:
    lines = []
    for turn in conversation:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_teacher_conversation(item: Dict[str, Any], response: str) -> List[Dict[str, str]]:
    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Problem: "
                + item.get("problem", "")
                + "\nReference Solution: "
                + item.get("reference_solution", "")
            ),
        },
    ]

    for entry in item.get("dialog_history", []):
        role = "assistant" if entry.get("user") in ["Teacher", "Tutor"] else "user"
        conversation.append({"role": role, "content": entry.get("text", "")})

    conversation.append({"role": "assistant", "content": response})
    return conversation


def normalize_preference_example(example: Dict[str, Any]) -> Dict[str, Any]:
    if "chosen" in example and "rejected" in example:
        return {"chosen": example["chosen"], "rejected": example["rejected"]}

    if "preferred" in example and "rejected" in example:
        return {"chosen": example["preferred"], "rejected": example["rejected"]}

    if "teacher_response_positive" in example and "teacher_response_negative" in example:
        return {
            "chosen": format_teacher_conversation(example, example["teacher_response_positive"]),
            "rejected": format_teacher_conversation(example, example["teacher_response_negative"]),
        }

    raise ValueError("Unsupported example format for QLoRA RM training.")


def render_conversation(tokenizer: AutoTokenizer, sample: Any) -> str:
    if isinstance(sample, list):
        try:
            return tokenizer.apply_chat_template(
                sample,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            return conversation_to_text(sample)

    if isinstance(sample, str):
        return sample

    raise ValueError(f"Unsupported conversation type: {type(sample)}")


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    normalized = dataset.map(normalize_preference_example)

    def tokenize_row(row: Dict[str, Any]) -> Dict[str, List[int]]:
        chosen_text = render_conversation(tokenizer, row["chosen"])
        rejected_text = render_conversation(tokenizer, row["rejected"])

        chosen_enc = tokenizer(
            chosen_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        rejected_enc = tokenizer(
            rejected_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }

    return normalized.map(
        tokenize_row,
        remove_columns=normalized.column_names,
        desc="Tokenizing preference pairs",
    )


@dataclass
class PairwiseDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        chosen_features = [
            {
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
            }
            for feature in features
        ]
        rejected_features = [
            {
                "input_ids": feature["rejected_input_ids"],
                "attention_mask": feature["rejected_attention_mask"],
            }
            for feature in features
        ]

        chosen_batch = self.tokenizer.pad(chosen_features, padding=True, return_tensors="pt")
        rejected_batch = self.tokenizer.pad(rejected_features, padding=True, return_tensors="pt")

        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
        }


class PairwiseRewardTrainer(Trainer):
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        chosen_rewards = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        ).logits.squeeze(-1)
        rejected_rewards = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        ).logits.squeeze(-1)

        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        if return_outputs:
            return loss, {
                "chosen_rewards": chosen_rewards,
                "rejected_rewards": rejected_rewards,
            }
        return loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return loss.detach(), None, None

        logits = torch.stack(
            [outputs["chosen_rewards"], outputs["rejected_rewards"]],
            dim=1,
        )
        labels = torch.ones(logits.size(0), dtype=torch.long, device=logits.device)
        return loss.detach(), logits.detach(), labels.detach()


def compute_metrics(eval_prediction: EvalPrediction) -> Dict[str, float]:
    if eval_prediction.predictions is None or len(eval_prediction.predictions) == 0:
        return {}

    predictions = np.asarray(eval_prediction.predictions)
    chosen_scores = predictions[:, 0]
    rejected_scores = predictions[:, 1]

    return {
        "preference_accuracy": float(np.mean(chosen_scores > rejected_scores)),
        "mean_chosen_score": float(np.mean(chosen_scores)),
        "mean_rejected_score": float(np.mean(rejected_scores)),
        "mean_margin": float(np.mean(chosen_scores - rejected_scores)),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=1,
        trust_remote_code=args.trust_remote_code,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, peft_config)

    dataset_splits = load_preference_splits(args)
    train_dataset = preprocess_dataset(dataset_splits["train"], tokenizer, args.max_length)
    eval_dataset = preprocess_dataset(dataset_splits["eval"], tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps" if len(eval_dataset) > 0 else "no",
        save_strategy="steps",
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        load_best_model_at_end=len(eval_dataset) > 0,
        metric_for_best_model="preference_accuracy" if len(eval_dataset) > 0 else None,
        greater_is_better=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
    )

    trainer = PairwiseRewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        processing_class=tokenizer,
        data_collator=PairwiseDataCollator(tokenizer),
        compute_metrics=compute_metrics if len(eval_dataset) > 0 else None,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if len(eval_dataset) > 0:
        metrics = trainer.evaluate()
        with open(Path(args.output_dir) / "final_eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Final eval metrics: %s", metrics)


if __name__ == "__main__":
    main()
