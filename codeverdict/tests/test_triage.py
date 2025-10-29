# codeverdict/tests/test_triage.py
import pytest
from codeverdict.evaluation.triage_engine import CodeVerdictTriageEngine
from codeverdict.data.models import CodeCompletion

def test_triage_engine_initialization():
    engine = CodeVerdictTriageEngine()
    assert engine.auto_eval_threshold == 0.8
    assert engine.manual_sample_rate == 0.5

def test_triage_decision_making():
    engine = CodeVerdictTriageEngine()
    completion = CodeCompletion(
        prompt_id="test_1", 
        model_id="test_model",
        completion="test code",
        metadata={"prompt_type": "code_generation"}
    )
    
    decision = engine._make_triage_decision(completion)
    assert decision in engine.TriageDecision