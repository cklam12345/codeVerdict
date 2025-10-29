# codeverdict/tests/test_evaluation.py
import pytest
from codeverdict.evaluation.auto_evaluator import CodeVerdictAutoEvaluator
from codeverdict.data.models import CodeCompletion

def test_auto_evaluator_initialization():
    evaluator = CodeVerdictAutoEvaluator()
    assert evaluator is not None

def test_code_quality_evaluation():
    evaluator = CodeVerdictAutoEvaluator()
    completion = CodeCompletion(
        prompt_id="test_1",
        model_id="test_model",
        completion="def hello():\n    return 'world'"
    )
    
    scores = evaluator.evaluate_code_quality(completion)
    assert "readability" in scores
    assert "security" in scores
    assert all(0 <= score <= 1 for score in scores.values())