# codeverdict/evaluation/auto_evaluator.py
import phoenix as px
import pandas as pd
from typing import List, Dict, Any, Tuple
import numpy as np
from codeverdict.data.models import CodeCompletion, CodeEvaluationCriteria

class CodeVerdictAutoEvaluator:
    def __init__(self):
        self.session = px.launch_app()
        self.security_patterns = [
            "eval(", "exec(", "pickle.loads", "os.system", "subprocess.call",
            "shell=True", "input()", "marshal.loads", "__import__", "compile("
        ]
        
    def evaluate_code_correctness(self, completion: CodeCompletion, test_cases: List[Dict]) -> Dict[str, float]:
        """Evaluate code correctness using test cases"""
        passed_tests = 0
        total_tests = len(test_cases)
        
        for test_case in test_cases:
            if self._simulate_test_execution(completion.completion, test_case):
                passed_tests += 1
                
        correctness_score = passed_tests / total_tests if total_tests > 0 else 0.0
        return {"correctness": correctness_score}
    
    def evaluate_code_quality(self, completion: CodeCompletion) -> Dict[str, float]:
        """Evaluate code quality using various metrics"""
        code = completion.completion
        
        readability_score = self._calculate_readability(code)
        efficiency_score = self._estimate_efficiency(code)
        security_score = self._check_security_issues(code)
        style_score = self._check_style_adherence(code)
        
        return {
            "readability": readability_score,
            "efficiency": efficiency_score, 
            "security": security_score,
            "style_adherence": style_score
        }
    
    def generate_verdict(self, completion: CodeCompletion, scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate final verdict based on evaluation scores"""
        overall_score = np.mean(list(scores.values()))
        
        # Determine verdict status based on scores
        if overall_score >= 0.8:
            status = "auto_approved"
            confidence = 0.9
        elif overall_score >= 0.6:
            status = "manual_review"
            confidence = 0.7
        else:
            status = "rejected"
            confidence = 0.8
            
        return {
            "completion_id": completion.id,
            "status": status,
            "overall_score": overall_score,
            "confidence": confidence,
            "criteria_scores": scores,
            "evaluated_at": pd.Timestamp.now().isoformat()
        }
    
    def _calculate_readability(self, code: str) -> float:
        """Calculate code readability score"""
        lines = code.split('\n')
        if len(lines) == 0:
            return 0.0
            
        # Calculate various readability metrics
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        comment_ratio = comment_lines / len(lines) if lines else 0
        
        # Combine metrics
        line_length_score = max(0, 1 - (avg_line_length - 40) / 100)
        readability = (line_length_score + min(comment_ratio * 3, 1)) / 2
        
        return min(1.0, max(0.0, readability))
    
    def _check_security_issues(self, code: str) -> float:
        """Check for common security issues"""
        score = 1.0
        issues_found = 0
        
        for pattern in self.security_patterns:
            if pattern in code:
                issues_found += 1
                score -= 0.15  # Penalize for each security issue
                
        # Additional security checks
        if "password" in code.lower() and "getpass" not in code:
            score -= 0.1
                
        return max(0.0, score)
    
    def _estimate_efficiency(self, code: str) -> float:
        """Estimate code efficiency based on patterns"""
        efficiency_indicators = [
            ("for loop in for loop", -0.3),
            ("while True", -0.2),
            ("recursion", -0.1),
            ("list comprehension", 0.1),
            ("generator", 0.2),
            ("built-in functions", 0.1)
        ]
        
        score = 0.7  # Base score
        
        for pattern, impact in efficiency_indicators:
            if pattern in code.lower():
                score += impact
                
        return max(0.0, min(1.0, score))
    
    def _check_style_adherence(self, code: str) -> float:
        """Check basic style adherence"""
        lines = code.split('\n')
        if not lines:
            return 0.0
            
        style_issues = 0
        
        # Check for very long lines
        for line in lines:
            if len(line) > 100:
                style_issues += 1
                
        # Check for proper indentation (basic check)
        if code and not code[0].isspace():
            style_issues += 1
            
        # Calculate score
        max_issues = max(len(lines) // 10, 1)
        style_score = 1.0 - (style_issues / max_issues)
        
        return max(0.0, min(1.0, style_score))
    
    def _simulate_test_execution(self, code: str, test_case: Dict) -> bool:
        """Simulate test execution (placeholder for actual execution)"""
        # In a real implementation, this would actually execute the code
        # For now, we'll simulate based on code patterns
        
        # Simple simulation logic
        test_input = test_case.get('input', '')
        expected_output = test_case.get('output', '')
        
        # Check if code contains expected patterns
        if 'return' in code and expected_output:
            return True  # Simulate passing test
            
        return False  # Simulate failing test