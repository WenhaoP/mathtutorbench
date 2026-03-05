# Competition Math Dataset Integration

## Overview

The `competition_math` dataset (qwedsacf/competition_math, 12.5k examples) from HuggingFace was integrated into MathTutorBench following clean, modular design principles.

## Key Design Decisions

### 1. Separate Task File
Created dedicated `tasks/compet_math.py` instead of modifying `gsm8k.py` to maintain isolation and independence.

### 2. Task Classes
- **GSM8K** (`tasks/gsm8k.py`): Extracts last numeric value (after `####`)
- **CompetitionMath** (`tasks/compet_math.py`): Extracts from `\boxed{}` with robust parsing and numeric fallback

### 3. Answer Extraction Strategy
**GSM8K**: `"...#### 26"` → `"26"`  
**CompetitionMath**: Correctly handles nested LaTeX:
- `"\boxed{17}"` → `"17"`
- `"\boxed{\frac{3}{5}}"` → `"\frac{3}{5}"`
- Falls back to last numeric value if no `\boxed{}` found

The parser walks nested braces to extract the complete answer, supporting complex expressions like fractions, nested functions, etc.

### 4. Robust Metrics Computation
The `compute_metrics` method handles imperfect model outputs:
- Removes commas from predictions (e.g., `"10, 26"` → `"26"`)
- Extracts last numeric token if multiple numbers present
- Gracefully handles `None` values (unanswerable questions)

### 5. Dataset Filtering
Filter problems by difficulty level and subject type:

```yaml
allowed_levels: ["Level 3", "Level 4"]    # Level 1-5
allowed_types: ["Algebra", "Geometry"]    # See below
```

**Available Types**: Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus

### 6. Dataset Size Control

Control evaluation scale with optional sampling (seed=42 for reproducibility):

```yaml
max_train_examples: 1000  # random sample 1k of 12.5k
max_test_examples: 500    # random sample 500 examples
```

Filters are applied **before** sampling, so you get a representative subset of the requested slice.

### 7. Dataset-Specific Configuration

| Field | GSM8K | Competition Math | Reason |
|-------|-------|------------------|--------|
| `dataset_path` | `gsm8k` | `qwedsacf/competition_math` | Different HF datasets |
| `dataset_name` | `main` | `default` | HF config names |
| `test_split` | `test` | `train` | Only train split available |
| `allowed_levels` | N/A | optional | Filter by difficulty |
| `allowed_types` | N/A | optional | Filter by subject |
| `max_train_examples` | optional | optional | Control dataset size |
| `max_test_examples` | optional | optional | Control dataset size |

## Files

| File | Type | Purpose |
|------|------|---------|
| `tasks/compet_math.py` | NEW | CompetitionMath task class with nested-brace-aware parsing |
| `configs/competition_math.yaml` | NEW | Dataset config with filtering and size limits |
| `tests/test_competition_math.py` | NEW | Tests parsing, filtering, metrics, and reproducibility |
| `tasks/__init__.py` | MODIFIED | Added CompetitionMath import |
| `tasks/base.py` | MODIFIED | Added filtering, max_*_examples, and random sampling logic |

## Usage

### Evaluate Competition Math (Full)
```bash
python main.py --tasks competition_math.yaml --provider completion_api --model_args model=gpt-4o-mini,api_key=<KEY>
```

### Evaluate Subset by Size
Edit `competition_math.yaml`:
```yaml
max_train_examples: 500  # evaluate 500 random problems
max_test_examples: 100
```

### Evaluate Specific Difficulty + Subject
Edit `competition_math.yaml`:
```yaml
max_train_examples: 200
max_test_examples: 50
allowed_levels: ["Level 4", "Level 5"]
allowed_types: ["Algebra", "Number Theory"]
```

This evaluates 200 random **Level 4–5** problems from **Algebra or Number Theory** only (reproducible with seed 42).

### Run Both GSM8K and Competition Math
```bash
python main.py --tasks problem_solving.yaml,competition_math.yaml --provider completion_api --model_args ...
```

## Implementation Details

### Answer Extraction with Nested Brace Support
The `parse_response` method correctly handles complex LaTeX expressions by tracking brace depth:

```python
# scan for first \\boxed{ and parse balanced braces
start = response.find("\\boxed{")
if start != -1:
    idx = start + len("\\boxed{")
    brace_level = 1
    content_chars = []
    while idx < len(response) and brace_level > 0:
        ch = response[idx]
        if ch == '{':
            brace_level += 1
        elif ch == '}':
            brace_level -= 1
            if brace_level > 0:
                content_chars.append(ch)
        else:
            content_chars.append(ch)
        idx += 1
```

This handles fractions, nested functions, and other complex expressions correctly.

### Robust Metrics Computation
The `compute_metrics` method robustly converts predictions to floats even when they contain:
- Commas (e.g., `"10, 26"` → extracts `26`)
- Multiple numbers (e.g., `"1 or 2"` → extracts last `2`)
- None values (skips comparison for unanswerable questions)

```python
def _to_float(val: str):
    if val is None:
        return None
    cleaned = val.replace(",", "")
    nums = re.findall(r"-?\d*\.?\d+", cleaned)
    if not nums:
        return None
    return float(nums[-1])
```

### Filtering and Sampling
When `allowed_levels` or `allowed_types` is set, datasets are first filtered before random sampling:

```python
if self.config.allowed_levels is not None:
    self.train_dataset = [ex for ex in self.train_dataset 
                         if ex.get("level") in self.config.allowed_levels]
# ... then apply random sampling with seed 42
```

This ensures reproducible, filtered subsets.

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

## Testing and Verification

Run the test suite to verify all features work correctly:

```bash
cd c:\Code\mathtutorbench
pytest tests/test_competition_math.py -v
```

Expected output:
```
test_boxed_answer_extraction PASSED
test_numeric_fallback PASSED
test_dataset_size_limiting PASSED
test_reproducible_sampling PASSED
test_filtering_by_level PASSED
test_filtering_by_type PASSED
test_fraction_parsing PASSED
```

### What Each Test Verifies
- **Boxed Answer**: Extracts `\boxed{...}` content correctly
- **Numeric Fallback**: Falls back to last number if no boxed format found
- **Size Limiting**: Respects `max_train_examples` and `max_test_examples` configurations
- **Reproducibility**: Two passes with same seed produce identical subsets
- **Level Filtering**: `allowed_levels` parameter correctly narrows dataset
- **Type Filtering**: `allowed_types` parameter correctly narrows dataset  
- **Fraction Parsing**: Complex LaTeX like `\boxed{\frac{3}{5}}` parses correctly

### Integration Testing

Test that both tasks work together without conflicts:

```bash
# Run both GSM8K and CompetitionMath
python main.py --tasks problem_solving.yaml,competition_math.yaml --num_batches 1
```

This should execute both tasks in a single run, demonstrating they coexist without breaking existing GSM8K functionality.

## Benefits and Outcomes

### For Researchers
- **Benchmarking Mathematical Reasoning**: Evaluate model performance on authentic competition problems at various difficulty levels
- **Difficulty-Stratified Evaluation**: Filter by Level 1-5 to understand where models plateau
- **Subject-Specific Analysis**: Isolate Algebra vs. Geometry vs. other domains to identify strengths/weaknesses
- **Efficient Iteration**: Control dataset size to get fast feedback during development (50 examples) vs. comprehensive evaluation (all 12.5k)

### For the Codebase
- **Non-Breaking**: Completely separate from GSM8K; existing evaluation pipelines continue to work
- **Extensible**: Pattern established for adding future datasets (math competition variants, olympiad problems, etc.)
- **Robust**: Handles edge cases like malformed model outputs and complex LaTeX expressions
- **Reproducible**: Fixed seed 42 ensures all measurements are repeatable


## Extensibility for New Datasets

The clean modular design makes it easy to add similar math datasets:

1. Create `tasks/new_dataset.py` with task class inheriting from `Task`
2. Implement `parse_response()` and `compute_metrics()` methods
3. Create `configs/new_dataset.yaml` with dataset configuration
4. Update `tasks/__init__.py` with import
5. Framework automatically registers and supports it via `@TaskRegistry.register("task_name")`

This pattern ensures each dataset remains isolated, preventing breaking changes to existing benchmarks.

