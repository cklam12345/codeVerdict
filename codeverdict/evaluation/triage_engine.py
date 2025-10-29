# codeverdict/evaluation/triage_engine.py
from typing import List, Dict, Any, Tuple
from codeverdict.data.models import CodeCompletion, VerdictStatus
from codeverdict.evaluation.auto_evaluator import CodeVerdictAutoEvaluator
import numpy as np
from enum import Enum

class TriageDecision(Enum):
    AUTO_APPROVE = "auto_approve"
    MANUAL_REVIEW = "manual_review"
    AUDIT_SAMPLE = "audit_sample"

class CodeVerdictTriageEngine:
    def __init__(self, auto_eval_threshold: float = 0.8, manual_sample_rate: float = 0.5):
        self.auto_eval_threshold = auto_eval_threshold
        self.manual_sample_rate = manual_sample_rate
        self.auto_evaluator = CodeVerdictAutoEvaluator()
        
    def triage_completions(self, completions: List[CodeCompletion]) -> Tuple[List[CodeCompletion], List[CodeCompletion]]:
        """Split completions into auto-evaluated and manual review batches"""
        auto_eval_batch = []
        manual_review_batch = []
        
        for completion in completions:
            decision = self._make_triage_decision(completion)
            
            if decision == TriageDecision.AUTO_APPROVE:
                auto_eval_batch.append(completion)
            elif decision == TriageDecision.MANUAL_REVIEW:
                manual_review_batch.append(completion)
            else:  # AUDIT_SAMPLE
                if np.random.random() < 0.5:
                    auto_eval_batch.append(completion)
                else:
                    manual_review_batch.append(completion)
        
        # Ensure we maintain approximately the desired 50/50 split
        total = len(completions)
        if total > 0:
            current_auto_ratio = len(auto_eval_batch) / total
            target_auto_ratio = 1 - self.manual_sample_rate
            
            if abs(current_auto_ratio - target_auto_ratio) > 0.1:
                self._rebalance_batches(auto_eval_batch, manual_review_batch, target_auto_ratio)
        
        print(f"🔀 CodeVerdict Triage: {len(auto_eval_batch)} auto-eval, {len(manual_review_batch)} manual review")
        return auto_eval_batch, manual_review_batch
    
    def _make_triage_decision(self, completion: CodeCompletion) -> TriageDecision:
        """Make triage decision for a single completion"""
        prompt_type = completion.metadata.get('prompt_type', 'code_generation')
        
        # Critical security prompts always go to manual review
        if prompt_type == "security_audit":
            return TriageDecision.MANUAL_REVIEW
            
        # High-risk bug fixes get careful review
        elif prompt_type == "bug_fixing":
            quality_scores = self.auto_evaluator.evaluate_code_quality(completion)
            overall_quality = np.mean(list(quality_scores.values()))
            
            if overall_quality > self.auto_eval_threshold:
                return TriageDecision.AUTO_APPROVE
            elif overall_quality < 0.5:
                return TriageDecision.MANUAL_REVIEW
            else:
                return TriageDecision.AUDIT_SAMPLE
                
        # Standard code generation with sampling
        else:
            if np.random.random() < self.manual_sample_rate:
                return TriageDecision.MANUAL_REVIEW
            else:
                return TriageDecision.AUTO_APPROVE
    
    def _rebalance_batches(self, auto_batch: List[CodeCompletion], manual_batch: List[CodeCompletion], target_auto_ratio: float):
        """Rebalance batches to maintain target ratio"""
        total = len(auto_batch) + len(manual_batch)
        target_auto_count = int(total * target_auto_ratio)
        
        current_auto_count = len(auto_batch)
        
        if current_auto_count > target_auto_count:
            move_count = current_auto_count - target_auto_count
            to_move = auto_batch[-move_count:]
            manual_batch.extend(to_move)
            del auto_batch[-move_count:]
        else:
            move_count = target_auto_count - current_auto_count
            to_move = manual_batch[-move_count:]
            auto_batch.extend(to_move)
            del manual_batch[-move_count:]