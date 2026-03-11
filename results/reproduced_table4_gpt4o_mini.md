# Reproduced Table 4 Results (GPT-4o-mini-2024-07-18)

This file summarizes the currently reproduced pedagogical evaluation results for `gpt-4o-mini-2024-07-18`.

## Pedagogical Reward-Model Results

| Model | Scaffolding Win Rate | Pedagogy IF Win Rate | Scaffolding (Hard) | Pedagogy IF (Hard) |
| --- | ---: | ---: | ---: | ---: |
| GPT-4o-mini-2024-07-18 | 0.5243 | 0.8435 | 0.4771 | 0.7278 |

## Additional Competition Math Result

| Model | Competition Math Accuracy |
| --- | ---: |
| GPT-4o-mini-2024-07-18 | 0.8860 |

## Source Artifacts

- Reward-model scoring JSONs were saved under `/content/drive/MyDrive/mathtutorbench_repro/scaffolding_scores/`
- Generation files were saved under `/content/drive/MyDrive/mathtutorbench_repro/results/`
- Competition Math result was saved under `/content/drive/MyDrive/mathtutorbench_repro/competition_math_results/`

## Notes

- These values come from the released pedagogical reward model (`eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel`).
- The upstream scoring script completed scoring successfully, but its final YAML update step failed for hyphenated model names, so the values were preserved from the generated `*_results.json` files instead.
