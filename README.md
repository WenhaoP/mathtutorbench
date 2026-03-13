# MathTutorBench: A Benchmark for Measuring Open-ended Pedagogical Capabilities of LLM Tutors
[![Arxiv](https://img.shields.io/badge/Arxiv-2502.18940-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2502.18940)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/deed.en)
[![Python Versions](https://img.shields.io/badge/Python-3.12-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

## Overview
**MathTutorBench** is a benchmark which provides a unified framework for evaluating open-ended pedagogical capabilities of large langauge models (LLMs) tutors across three high level teacher skills and seven concrete tasks.

This repository is a fork of the original project and includes additional experiments and reproduction scripts beyond the original release.


## Key Features
- **Automatic Evaluation**: The benchmark is designed to be run automatically on any new models you are developing.
- **Comprehensive Metrics**: The benchmark covers a three high level tasks skills and seven tasks to evaluate in the domain of math tutoring.
- **Teacher-Grounded Evaluation**: Each task is annotated with teacher ground truths and compared to it.
- **Fast execution loop**: Run benchmark on different tasks very quickly.

<p align="center">
<img src="./images/skills.png" alt="Skills" width="400">
</p>

## Quick Start - Evaluate a New Model
### 0. Run your model locally using vllm - skip if you are using API
For more details on how to run your model locally using vllm, see [vllm](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#vllm-server) documentation. Optionally add tensor parallelism if you have multiple GPUs and your model is large.
```bash
vllm serve [[model_name]] --seed 42 --tensor-parallel-size 4
```

### 1. Run task(s) from the benchmark
```bash
# Example with vllm model
python main.py --tasks mistake_location.yaml --provider completion_api --model_args base_url=http://localhost:8000/v1,model=meta-llama/Llama-3.2-3B-Instruct
# Example with OpenAI API
python main.py --tasks mistake_correction.yaml --provider completion_api --model_args model=gpt-4o-mini-2024-07-18,api_key=<API_KEY>
# Example with LearnLM Gemini API
python main.py --tasks student_solution_correctness.yaml --provider gemini --model_args model==learnlm-1.5-pro-experimental,api_key=<API_KEY>

```
- Required:
  - `--tasks`: Task definition file in the `configs` folder. Use comma `,` separated list for multiple sequential tasks.
    - `problem_solving.yaml`: Task definition for problem solving.
    - `socratic_questioning.yaml`: Task definition for socratic questioning.
    - `student_solution_correctness.yaml`: Task definition for student solution generation.
    - `mistake_location.yaml`: Task definition for mistake location.
    - `mistake_correction.yaml`: Task definition for mistake correction.
    - `scaffolding_generation.yaml`: Task definition for scaffolding generation.
    - `pedagogy_following.yaml`: Task definition for pedagogy following.
    - `scaffolding_generation_hard.yaml`: Task definition for scaffolding generation hard.
    - `pedagogy_following_hard.yaml`: Task definition for pedagogy following hard.
  - `--provider`: API provider to use for the task.
    - `completion_api`: Use the completion API for the task. Support any OpenAI-type API. Use for openai and vllm models.
    - `gemini`: Use the gemini API for the task. 
  - `--model_args`: Model arguments to pass to the API provider.
    - `base_url`: Base URL of the API provider. Empty for openai and gemini.
    - `model`: Model name to use for the task. Default is the first available model.
    - `api_key`: API key to access API. Empty for vllm models.
    - `is_chat`: Whether the requests to the model should use chat-based template (Chat Completion API). Default is False.
    - `temperature`: Temperature for sampling. Default is 0.0.
    - `max_tokens`: Maximum tokens to generate. Default is 2048.
    - `max_retries`: Maximum retries for the API. Default is 3.

The performance of different benchmarked models averaged across tasks for Qwen2.5 family is as follows (using vllm version 0.8.0 on one node with 4x GH200 GPUs):

| Model                  | Total time [min] | Examples/sec | Tokens/sec |
|-------------------------|------------------|--------------|------------|
| Qwen2.5-1.5B-Instruct  | 61.1             | 2.73         | 757.6      |
| Qwen2.5-7B-Instruct    | 58.3             | 2.86         | 1012       |
| Qwen2.5-32B-Instruct   | 545.3            | 0.31         | 166.3      |
| Qwen2.5-72B-Instruct   | 233.9            | 0.71         | 135.2      |


### 2. Run reward model of the Pedagogical Ability tasks
Set the `--data_path` to model outputs of the pedagogical ability tasks. The model computes win rates of generated teacher utterance over the ground truth teacher utterance.
```bash
python reward_model/compute_scaffolding_score.py --data_path results/generations-<specific-model>.json
```

As the model is small in size (1.5B parameters), running the full evaluation should be fast (within 10 minutes on a single GPU).
Reward model computation performance with different batch sizes on a single GH200 GPU:

| Batch size | Total time [sec] | Examples/sec | Tokens/sec |
|------------|------------------|--------------|------------|
| 1          | 419.58           | 7.01         | 6928.0     |
| 8          | 406.08           | 7.25         | 7159.3     |
| 64         | 413.28           | 7.12         | 7034.8     |
| 128        | 408.87           | 7.20         | 7110.0     |



### 3. Visualize results
Results are available in the `results` folder. To visualize the results, run:
```bash
python visualize.py --results_dir results/
```

<img src="./images/figure2.png" alt="Skills" width="800">


### 4. Reproducing Table 4 (standalone script)
`run_table4.py` runs the full Table 4 experiment locally: all tasks for LLaMA 3.2 3B, LLaMA 3.1 8B, Qwen2.5-Math-7B, Qwen2.5-7B, Qwen2-Math-7B, Qwen2-7B, and SocraticLM, then the pedagogical reward model, and prints the table. Run from the repo root (e.g. `math-main`). Requires `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for gated models.

```bash
export HF_TOKEN=your_token   # for gated models (Llama, etc.)
python run_table4.py                     # full run
python run_table4.py --max-examples 20   # quick run (cap examples per task)
python run_table4.py --no-cache          # ignore cached results
```

Results are cached in `cached_table4_rows/` so re-runs reuse completed models/tasks.

## Reproducibility instructions (this fork)

This fork includes focused reproducibility instructions and example commands to reproduce the experiments from the original paper and runs included in this repository. The repository already contains task definitions, dataloaders, example scripts and a reward-model training/evaluation pipeline. If you need additional help reproducing a specific experiment, open an issue with the experiment name.

Required items and commands to reproduce experiments in this repository:

- **Dependencies:** Install the Python requirements with:

```bash
pip install -r requirements.txt
```

- **Data download (where applicable):**
  - The repository includes example dataset files under the `datasets/` folder (e.g., `datasets/mathdial_bridge*.json`).
  - The pedagogical reward-model dataset is available on Hugging Face at [`dmacjam/pedagogical-rewardmodel-data`](https://huggingface.co/datasets/dmacjam/pedagogical-rewardmodel-data). Download it with the `datasets` library:

- **Preprocessing code + command:**
  - Preprocessing and data loading are implemented in the `dataloaders/` package (`dataloaders/base.py`, `dataloaders/mathbridge.py`) and integrated into task runners. For most workflows the preprocessing step is already handled when you run `main.py` for a task. Example task run (which will invoke the repo's data loading logic):

  ```bash
  python main.py --tasks mistake_correction.yaml --provider completion_api --model_args model=gpt-4o-mini-2024-07-18,api_key=<API_KEY>
  ```

  - If you add a custom preprocessing pipeline, place it under `dataloaders/` or provide a small CLI wrapper that outputs processed JSON that the `tasks` expect.

- **Training code + command:**
  - Reward-model training is available in `reward_model/train_reward_model.py`. To run training (example):

  ```bash
  python reward_model/train_reward_model.py
  ```

  - See `reward_model/README.md` for additional training hyperparameters and configuration options.

- **Evaluation code + command:**
  - Compute the pedagogical/Scaffolding win-rate (reward-model evaluation) with:

  ```bash
  python reward_model/compute_scaffolding_score.py --data_path results/generations-<model>.json
  ```

  - To run tasks and produce model outputs (used as input to evaluation), use `main.py` as shown above. See the top of this README for example task runs.

- **Pretrained models/checkpoints:**
  - This repository (and the original repository) does not include large pretrained model checkpoints.


## Citation
Please cite as:
```bibtex
@inproceedings{macina-etal-2025-mathtutorbench,
    title = "{M}ath{T}utor{B}ench: A Benchmark for Measuring Open-ended Pedagogical Capabilities of {LLM} Tutors",
    author = "Macina, Jakub  and
      Daheim, Nico  and
      Hakimi, Ido  and
      Kapur, Manu  and
      Gurevych, Iryna  and
      Sachan, Mrinmaya",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.11/",
    doi = "10.18653/v1/2025.emnlp-main.11",
    pages = "204--221",
    ISBN = "979-8-89176-332-6",
```

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
