#!/usr/bin/env python3
import argparse
import gc
import json
import logging
import os
import shutil
import sys
import yaml
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

TASK_CONFIGS = [
    "problem_solving.yaml",
    "socratic_questioning.yaml",
    "student_solution_correctness.yaml",
    "mistake_location.yaml",
    "mistake_correction.yaml",
    "scaffolding_generation.yaml",
    "pedagogy_following.yaml",
    "scaffolding_generation_hard.yaml",
    "pedagogy_following_hard.yaml",
]
TASK_TO_METRIC = {
    "problem_solving": "accuracy_flexible-extract",
    "socratic_questioning": "bleu",
    "solution_correctness": "f1",
    "student_solution_correctness": "f1",
    "mistake_location": "f1_micro",
    "mistake_correction": "accuracy",
    "scaffolding_generation": "win_rate",
    "pedagogy_following": "win_rate",
    "scaffolding_generation_hard": "win_rate",
    "pedagogy_following_hard": "win_rate",
}
PEDAGOGY_TASKS = {"scaffolding_generation", "pedagogy_following", "scaffolding_generation_hard", "pedagogy_following_hard"}
TABLE4_COLUMNS = [
    "Problem solving", "Socratic", "Solution", "Mistake location", "Mistake correction",
    "scaff.", "ped.IF", "scaff. [hard]", "ped.IF [hard]",
]
COMPETITION_MATH_COLUMN = "Competition Math"
ALL_COLUMNS = TABLE4_COLUMNS + [COMPETITION_MATH_COLUMN]
TASK_TO_COL = {
    "problem_solving": "Problem solving",
    "socratic_questioning": "Socratic",
    "solution_correctness": "Solution",
    "student_solution_correctness": "Solution",
    "mistake_location": "Mistake location",
    "mistake_correction": "Mistake correction",
    "scaffolding_generation": "scaff.",
    "pedagogy_following": "ped.IF",
    "scaffolding_generation_hard": "scaff. [hard]",
    "pedagogy_following_hard": "ped.IF [hard]",
}
MODELS_LOAD_IN_8BIT = set()
REWARD_MODEL_ID = "eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel"
COMPETITION_MATH_CONFIG = "competition_math.yaml"


def apply_reward_model_patch():
    n = chr(10)
    _old_yaml = "    results_yaml_file = \"../results/\" + f\"results-{model_name}.yaml\"" + n + "    with open(results_yaml_file, 'r') as f:" + n + "        data = yaml.safe_load(f)" + n + n + "    if not data:" + n + "        data = {}"
    _new_yaml = "    results_yaml_file = Path(output_dir) / f\"results-{model_name}.yaml\"" + n + "    if results_yaml_file.exists():" + n + "        with open(results_yaml_file, 'r') as f:" + n + "            data = yaml.safe_load(f)" + n + "    else:" + n + "        data = {}" + n + "    if not data:" + n + "        data = {}"
    f = REPO_ROOT / "reward_model" / "compute_scaffolding_score.py"
    if not f.is_file():
        return
    os.system(f"sed -i 's/self\\.model(inputs)/self.model(**inputs)/' {f}")
    os.system(f"sed -i 's/Path(args\\.data_path)/Path(data_path)/' {f}")
    with open(f, "r") as fp:
        text = fp.read()
    if _old_yaml in text:
        text = text.replace(_old_yaml, _new_yaml)
    elif '"../results/"' in text or "../results/" in text:
        text = text.replace('    results_yaml_file = "../results/" + f"results-{model_name}.yaml"', '    results_yaml_file = Path(output_dir) / f"results-{model_name}.yaml"')
        old_block = "    with open(results_yaml_file, 'r') as f:" + n + "        data = yaml.safe_load(f)" + n + n + "    if not data:" + n + "        data = {}"
        new_block = "    if results_yaml_file.exists():" + n + "        with open(results_yaml_file, 'r') as f:" + n + "            data = yaml.safe_load(f)" + n + "    else:" + n + "        data = {}" + n + "    if not data:" + n + "        data = {}"
        text = text.replace(old_block, new_block)
    with open(f, "w") as fp:
        fp.write(text)


def _apply_stop(text: str, stop: list) -> str:
    if not stop:
        return text.strip()
    first = len(text)
    for s in stop:
        idx = text.find(s)
        if idx >= 0:
            first = min(first, idx)
    return text[:first].strip()


class LocalLlamaWrapper:
    def __init__(self, model_id: str, max_tokens: int = 2048, load_in_8bit: bool = False):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if load_in_8bit and self.device == "cuda":
            import transformers.modeling_utils as _modeling_utils
            _orig_load = _modeling_utils.load_state_dict
            def _load_no_mmap(f, map_location="cpu", weights_only=True):
                return torch.load(f, map_location=map_location, weights_only=weights_only, mmap=False)
            _modeling_utils.load_state_dict = _load_no_mmap
            try:
                from transformers import BitsAndBytesConfig
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                    device_map="auto",
                    trust_remote_code=True,
                )
            finally:
                _modeling_utils.load_state_dict = _orig_load
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
            )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, messages, system_prompt: str, stop=None):
        chat = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Please provide your response."}]
        text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        reply = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return _apply_stop(reply, stop or [])


LOCAL_LLM_CACHE = {}


def get_local_llm(model_id: str):
    global LOCAL_LLM_CACHE
    if model_id not in LOCAL_LLM_CACHE:
        load_8bit = model_id in MODELS_LOAD_IN_8BIT
        LOCAL_LLM_CACHE[model_id] = LocalLlamaWrapper(model_id, load_in_8bit=load_8bit)
    return LOCAL_LLM_CACHE[model_id]


def load_task_config(config_path: str):
    from tasks.base import TaskConfig
    with open(REPO_ROOT / "configs" / config_path, "r") as f:
        return TaskConfig(**yaml.safe_load(f))


def run_model_on_tasks(model_spec: dict, max_examples: int):
    from models.completion_api import create_llm_model, LLMConfig
    from registry import TaskRegistry

    provider = model_spec["provider"]
    model_name = model_spec["model"]
    if provider == "local":
        model = get_local_llm(model_name)
    else:
        config = LLMConfig(provider=provider, model=model_name, api_key=model_spec.get("api_key"), base_url=model_spec.get("base_url"), temperature=0.0, max_tokens=2048, is_chat=model_spec.get("is_chat", True))
        model = create_llm_model(config)
    safe_name = model_name.replace("/", "-").replace(".", "_")
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    task_metrics = {}
    for config_path in TASK_CONFIGS:
        task_config = load_task_config(config_path)
        task_name = task_config.name
        task_cls = TaskRegistry.get_task(task_name)
        task = task_cls(task_config)
        examples = task.get_test_examples()
        if max_examples is not None:
            examples = examples[:max_examples]
        predictions, targets, all_generations = [], [], []
        for example in tqdm(examples, desc=f"{model_name} / {task_name}"):
            example["shots"] = task_config.few_shot_samples
            response = model.generate(messages=[], system_prompt=task.get_system_prompt(example), stop=task_config.stop)
            pred = task.parse_response(response)
            predictions.append(pred)
            targets.append(task.format_ground_truth(example))
            if task_name in PEDAGOGY_TASKS:
                all_generations.append({
                    "problem": example.get("question", ""),
                    "reference_solution": example.get("reference_solution", "N/A"),
                    "dialog_history": example.get("conversation_json", []),
                    "dialog_formatted": example.get("dialog_history", ""),
                    "ground_truth_response": example.get("ground_truth_response", ""),
                    "generated_teacher_utterance": pred,
                })
        metrics = task.compute_metrics(predictions, targets)
        if task_name in PEDAGOGY_TASKS:
            gen_path = results_dir / f"generations-{safe_name}-{task_name}.json"
            with open(gen_path, "w") as f:
                json.dump(all_generations, f, indent=2)
            task_metrics[task_name] = {"generations_path": str(gen_path), **metrics}
        else:
            key = TASK_TO_METRIC.get(task_name) or list(metrics.keys())[0]
            task_metrics[task_name] = metrics.get(key, metrics)
    return task_metrics, safe_name


def run_reward_model_on_generations(task_metrics: dict, safe_name: str) -> dict:
    logging.getLogger("transformers").setLevel(logging.WARNING)
    from reward_model.compute_scaffolding_score import evaluate_preference_accuracy

    for task_name in PEDAGOGY_TASKS:
        info = task_metrics.get(task_name)
        if not info or not isinstance(info, dict) or "generations_path" not in info:
            continue
        path = info["generations_path"]
        if not Path(path).exists():
            continue
        res = evaluate_preference_accuracy(REWARD_MODEL_ID, path, batch_size=8, output_dir=str(REPO_ROOT / "scaffolding_scores"))
        task_metrics[task_name] = res.get("win_rate", 0.0)
    return task_metrics


def task_metrics_to_row(task_metrics: dict) -> dict:
    row = {col: "" for col in TABLE4_COLUMNS}
    for task_name, val in task_metrics.items():
        col = TASK_TO_COL.get(task_name)
        if col is None:
            continue
        if isinstance(val, dict):
            key = TASK_TO_METRIC.get(task_name, "win_rate")
            val = val.get(key, val.get("win_rate", ""))
        row[col] = round(val, 2) if isinstance(val, (int, float)) else val
    return row


def run_competition_math_single(model_spec, max_examples: int):
    from models.completion_api import create_llm_model, LLMConfig
    from registry import TaskRegistry

    provider = model_spec["provider"]
    model_name = model_spec["model"]
    if provider == "local":
        model = get_local_llm(model_name)
    else:
        config = LLMConfig(provider=provider, model=model_name, api_key=model_spec.get("api_key"), base_url=model_spec.get("base_url"), temperature=0.0, max_tokens=2048, is_chat=model_spec.get("is_chat", True))
        model = create_llm_model(config)
    task_config = load_task_config(COMPETITION_MATH_CONFIG)
    task = TaskRegistry.get_task(task_config.name)(task_config)
    examples = task.get_test_examples()
    if max_examples is not None:
        examples = examples[:max_examples]
    predictions, targets = [], []
    for example in tqdm(examples, desc=f"{model_name} / competition_math"):
        example["shots"] = task_config.few_shot_samples
        response = model.generate(messages=[], system_prompt=task.get_system_prompt(example), stop=task_config.stop)
        predictions.append(task.parse_response(response))
        targets.append(task.format_ground_truth(example))
    return task.compute_metrics(predictions, targets).get("accuracy", 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-examples", type=int, default=None, help="Cap examples per task (default: full)")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached rows")
    args = parser.parse_args()
    max_examples = args.max_examples
    use_cache = not args.no_cache

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token and not hf_token.startswith("hf_xxxx"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        try:
            from huggingface_hub import login
            login(token=hf_token)
        except Exception:
            pass

    apply_reward_model_patch()

    from tasks.compet_math import CompetitionMath

    MODELS_TO_RUN = [
        {"name": "LLaMA3.2-3B-Instruct", "provider": "local", "model": "meta-llama/Llama-3.2-3B-Instruct"},
        {"name": "LLaMA3.1-8B-Instruct", "provider": "local", "model": "meta-llama/Llama-3.1-8B-Instruct"},
        {"name": "Qwen2.5-Math-7B-Instruct", "provider": "local", "model": "Qwen/Qwen2.5-Math-7B-Instruct"},
        {"name": "Qwen2.5-7B-Instruct", "provider": "local", "model": "Qwen/Qwen2.5-7B-Instruct"},
        {"name": "Qwen2-Math-7B-Instruct", "provider": "local", "model": "Qwen/Qwen2-Math-7B-Instruct"},
        {"name": "Qwen2-7B-Instruct", "provider": "local", "model": "Qwen/Qwen2-7B-Instruct"},
        {"name": "SocraticLM", "provider": "local", "model": "CogBase-USTC/SocraticLM"},
    ]

    CACHE_DIR = REPO_ROOT / "cached_table4_rows"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_n = 100 if max_examples is None else max_examples
    FALLBACK_CACHE_DIR = (REPO_ROOT.parent / "mathtutorbench" / "cached_table4_rows").resolve()
    all_results = []

    for model_spec in MODELS_TO_RUN:
        display_name = model_spec["name"]
        model_id = model_spec["model"]
        safe_name = model_id.replace("/", "-").replace(".", "_")
        cache_path = CACHE_DIR / f"{safe_name}_n{cache_n}.json"
        _candidate_names = [f"{safe_name}_n{k}.json" for k in (cache_n, 300, "full")]
        row = {}
        if use_cache:
            for _name in _candidate_names:
                _p = CACHE_DIR / _name
                if _p.exists():
                    with open(_p, "r") as f:
                        row = json.load(f)
                    break
            if not row and FALLBACK_CACHE_DIR.exists():
                for _name in _candidate_names:
                    _fallback = FALLBACK_CACHE_DIR / _name
                    if _fallback.exists():
                        _dest = CACHE_DIR / _name
                        shutil.copy2(_fallback, _dest)
                        with open(_dest, "r") as f:
                            row = json.load(f)
                        break
        row["Model"] = display_name
        missing = [c for c in ALL_COLUMNS if c not in row or row.get(c) == ""]
        if not missing:
            all_results.append(row)
            continue
        if set(missing) == {COMPETITION_MATH_COLUMN}:
            row[COMPETITION_MATH_COLUMN] = round(run_competition_math_single(model_spec, max_examples), 2)
            with open(cache_path, "w") as f:
                json.dump({k: v for k, v in row.items() if k != "Model"}, f, indent=2)
            all_results.append(row)
            if model_id in LOCAL_LLM_CACHE:
                del LOCAL_LLM_CACHE[model_id]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        task_metrics, _ = run_model_on_tasks(model_spec, max_examples)
        task_metrics = run_reward_model_on_generations(task_metrics, safe_name)
        row_part = task_metrics_to_row(task_metrics)
        for k in TABLE4_COLUMNS:
            row[k] = row_part[k]
        if COMPETITION_MATH_COLUMN in missing:
            row[COMPETITION_MATH_COLUMN] = round(run_competition_math_single(model_spec, max_examples), 2)
        with open(cache_path, "w") as f:
            json.dump({k: v for k, v in row.items() if k != "Model"}, f, indent=2)
        all_results.append(row)
        if model_id in LOCAL_LLM_CACHE:
            del LOCAL_LLM_CACHE[model_id]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        import pandas as pd
        df = pd.DataFrame(all_results)
        cols = ["Model"] + [c for c in ALL_COLUMNS if c in (df.columns)]
        print(df[[c for c in cols if c in df.columns]].to_string())
    except ImportError:
        for r in all_results:
            print(r)


if __name__ == "__main__":
    main()
