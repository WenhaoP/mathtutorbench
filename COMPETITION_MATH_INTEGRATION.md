# Competition Math Dataset Integration

## Overview

The `competition_math` dataset (qwedsacf/competition_math, 12.5k examples) from HuggingFace was integrated into MathTutorBench following clean, modular design principles.

## Key Design Decisions

### 1. Separate Task File
Created dedicated `tasks/compet_math.py` instead of modifying `gsm8k.py` to maintain isolation and independence.

### 2. Task Classes
- **GSM8K** (`tasks/gsm8k.py`): Extracts last numeric value (after `####`)
- **CompetitionMath** (`tasks/compet_math.py`): Extracts from `\boxed{}` with numeric fallback

### 3. Answer Extraction
**GSM8K**: `"...#### 26"` → `"26"`  
**CompetitionMath**: `"...\boxed{17}..."` → `"17"` or fallback to last number

### 4. Dataset-Specific Configuration

| Field | GSM8K | Competition Math | Reason |
|-------|-------|------------------|--------|
| `dataset_path` | `gsm8k` | `qwedsacf/competition_math` | Different HF datasets |
| `dataset_name` | `main` | `default` | HF config names |
| `test_split` | `test` | `train` | Only train split available |
| `max_train_examples` | (optional) | (optional) | Control dataset size |

### 5. Dataset Size Control

Add to any config to limit dataset size with reproducible random sampling (seed=42):

```yaml
max_train_examples: 1000  # optional: random sample 1k of 12.5k
max_test_examples: 500    # optional: random sample 500 examples
```

Sampling is reproducible across runs using fixed seed 42—no manual control needed.

## Files

| File | Type | Purpose |
|------|------|---------|
| `tasks/compet_math.py` | NEW | CompetitionMath task class |
| `configs/competition_math.yaml` | NEW | Dataset config with optional size limits |
| `tests/test_competition_math.py` | NEW | Tests parsing, fallback, and reproducible sampling |
| `tasks/__init__.py` | MODIFIED | Added CompetitionMath import |
| `tasks/base.py` | MODIFIED | Added max_*_examples fields and random sampling logic |

## Usage

### Evaluate Competition Math
```bash
python main.py --tasks competition_math.yaml --provider completion_api --model_args model=gpt-4o-mini,api_key=<KEY>
```

### Evaluate Subset (100 examples)
Edit `competition_math.yaml`:
```yaml
max_train_examples: 100
```

### Run Both GSM8K and Competition Math
```bash
python main.py --tasks problem_solving.yaml,competition_math.yaml --provider completion_api --model_args ...
```

## Implementation Details

### Random Sampling
When `max_train_examples` or `max_test_examples` is set, `Task._load_dataset()` performs random subsampling:

```python
if self.config.max_train_examples is not None:
    import random
    random.seed(42)
    self.train_dataset = random.sample(
        self.train_dataset,
        min(self.config.max_train_examples, len(self.train_dataset))
    )
```

Reproducible, representative subset without requiring user to specify seed manually.

### Metrics Computation
Both tasks compute accuracy:
```python
correct = sum(1 for p, t in zip(predictions, targets)
              if p is not None and abs(p - t) < 1e-6)
accuracy = correct / total
```

## Testing

```bash
python tests/test_competition_math.py
```

Verifies:
- Boxed answer extraction
- Numeric fallback
- Dataset size limiting
- Reproducible sampling (two instantiations produce identical subsets)

## Key Features

✓ No breaking changes to GSM8K  
✓ Clean modular design (separate files)  
✓ Dataset-native answer formats  
✓ Reproducible random sampling  
✓ Optional dataset size control  
✓ Well-tested functionality  

## Future Extensibility

Easy to add new math datasets:
1. Create `tasks/new_dataset.py` with task class
2. Create `configs/new_dataset.yaml` with config
3. Update `tasks/__init__.py` with import
4. Framework automatically registers and supports it

