from typing import Dict, List
import re
from registry import TaskRegistry
from .base import Task


@TaskRegistry.register("problem_solving_competition_math")
class CompetitionMath(Task):
    """Task for evaluating problem solving on the Competition Math dataset.
    
    The Competition Math dataset uses LaTeX \boxed{...} to mark final answers,
    unlike GSM8K which uses the #### marker.
    """

    def parse_response(self, response: str) -> str:
        """Extract the final answer from the model's response.

        Attempts to extract the answer from a LaTeX \boxed{...} expression.
        Unlike the naive regex, this version correctly handles nested braces
        (e.g. `\boxed{\frac{3}{5}}`).
        If no boxed expression is found we fall back to extracting the
        last numeric token as before.
        """
        # scan for first occurrence of \boxed{ and then parse balanced braces
        start = response.find("\\boxed{")
        if start != -1:
            # position of opening brace
            idx = start + len("\\boxed{")
            brace_level = 1
            content_chars = []
            while idx < len(response) and brace_level > 0:
                ch = response[idx]
                if ch == '{':
                    brace_level += 1
                    content_chars.append(ch)
                elif ch == '}':
                    brace_level -= 1
                    if brace_level > 0:
                        content_chars.append(ch)
                else:
                    content_chars.append(ch)
                idx += 1
            if brace_level == 0:
                return ''.join(content_chars).strip()
            # otherwise fall through if braces unbalanced

        # Fall back to last numeric value
        numbers = re.findall(r'-?\d*\.?\d+', response)
        if numbers:
            return str(numbers[-1])
        return None

    def compute_metrics(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """Compute accuracy metrics for Competition Math task.

        Unlike GSM8K, predictions may contain commas or extra numbers (e.g.
        "10, 26").  We extract the last numeric token from each string and
        convert that to a float before comparing with the target.
        """
        results = {}

        def _to_float(val: str):
            if val is None:
                return None
            # remove commas then grab last number via regex
            import re
            cleaned = val.replace(",", "")
            nums = re.findall(r"-?\d*\.?\d+", cleaned)
            if not nums:
                return None
            try:
                return float(nums[-1])
            except ValueError:
                return None

        processed_predictions = [_to_float(p) for p in predictions]
        numeric_targets = [_to_float(t) for t in targets]

        correct = sum(
            1 for p, t in zip(processed_predictions, numeric_targets)
            if p is not None and t is not None and abs(p - t) < 1e-6
        )
        total = len(numeric_targets)
        accuracy = correct / total if total > 0 else 0.0

        results["accuracy"] = accuracy
        return results
