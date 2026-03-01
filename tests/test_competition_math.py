"""Test the competition_math problem solving task."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from tasks.base import TaskConfig
from registry import TaskRegistry


def test_competition_math_loading():
    """Test that the competition_math task can be loaded and initialized."""
    # Load the config
    with open("configs/competition_math.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = TaskConfig(**config_dict)
    
    # Get and instantiate the task
    task_cls = TaskRegistry.get_task(config.name)
    task = task_cls(config)
    
    # Test the parse_response method with a boxed answer
    response = r"The answer is \boxed{42}"
    parsed = task.parse_response(response)
    assert parsed == "42", f"Expected '42', got {parsed}"
    
    # Test with numeric fallback
    response = "The answer is 17"
    parsed = task.parse_response(response)
    assert parsed == "17", f"Expected '17', got {parsed}"

    # verify that limiting examples works
    # reload with a small max_train_examples and max_test_examples
    config_dict_limited = dict(config_dict)
    config_dict_limited['max_train_examples'] = 3
    config_dict_limited['max_test_examples'] = 2
    limited_config = TaskConfig(**config_dict_limited)
    limited_task = task_cls(limited_config)
    assert len(limited_task.train_dataset) == 3
    assert len(limited_task.test_dataset) == 2

    # check reproducibility: instantiate again and compare
    limited_task2 = task_cls(limited_config)
    assert limited_task.train_dataset == limited_task2.train_dataset
    assert limited_task.test_dataset == limited_task2.test_dataset

    # ensure complex boxed answers are parsed correctly
    frac_response = r"The answer is \boxed{\frac{3}{5}}"
    assert task.parse_response(frac_response) == "\\frac{3}{5}"

    # verify filtering by level/type works
    filter_cfg = dict(config_dict)
    filter_cfg['allowed_levels'] = ['Level 5']
    filter_cfg['allowed_types'] = ['Algebra']
    filter_cfg['max_train_examples'] = 10
    filter_cfg['max_test_examples'] = 10
    filtered_task = task_cls(TaskConfig(**filter_cfg))
    # every example should have the requested level/type
    for ex in filtered_task.train_dataset + filtered_task.test_dataset:
        assert ex.get('level') == 'Level 5'
        assert ex.get('type') == 'Algebra'

    print("✓ All tests passed!")


if __name__ == "__main__":
    test_competition_math_loading()
