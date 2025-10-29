# codeverdict/utils/helpers.py
import json
from typing import Dict, Any, List
from datetime import datetime

def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate Pass@k metric
    
    Args:
        n: Total number of samples
        c: Number of correct samples  
        k: k for Pass@k
        
    Returns:
        Pass@k score
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def cluster_failures_by_pattern(failures: List[Dict]) -> Dict[str, List]:
    """Cluster failures by common patterns"""
    clusters = {
        "syntax_errors": [],
        "logical_errors": [], 
        "security_issues": [],
        "efficiency_problems": [],
        "style_violations": []
    }
    
    for failure in failures:
        scores = failure.get("criteria_scores", {})
        
        if scores.get("security", 1) < 0.5:
            clusters["security_issues"].append(failure)
        elif scores.get("correctness", 1) < 0.5:
            clusters["logical_errors"].append(failure)
        elif scores.get("efficiency", 1) < 0.5:
            clusters["efficiency_problems"].append(failure)
        elif scores.get("style_adherence", 1) < 0.5:
            clusters["style_violations"].append(failure)
        else:
            clusters["syntax_errors"].append(failure)
            
    return clusters

def generate_improvement_report(verdicts: List[Dict], model_id: str) -> Dict[str, Any]:
    """Generate improvement report from verdicts"""
    failed_verdicts = [v for v in verdicts if v["overall_score"] < 0.7]
    clusters = cluster_failures_by_pattern(failed_verdicts)
    
    return {
        "model_id": model_id,
        "total_evaluations": len(verdicts),
        "success_rate": len([v for v in verdicts if v["overall_score"] >= 0.7]) / len(verdicts),
        "failure_clusters": {
            cluster: len(items) for cluster, items in clusters.items()
        },
        "recommendations": _generate_recommendations(clusters),
        "generated_at": datetime.now().isoformat()
    }

def _generate_recommendations(clusters: Dict) -> List[str]:
    """Generate improvement recommendations from failure clusters"""
    recommendations = []
    
    if clusters["security_issues"]:
        recommendations.append("Focus on security training with sanitized input examples")
    if clusters["logical_errors"]:
        recommendations.append("Add more edge case testing and validation examples") 
    if clusters["efficiency_problems"]:
        recommendations.append("Include algorithmic complexity and optimization training")
    if clusters["syntax_errors"]:
        recommendations.append("Reinforce language syntax and basic constructs")
        
    return recommendations