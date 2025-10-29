# codeverdict/orchestration/workflows.py
from typing import List, Dict, Any
import asyncio
from codeverdict.data.models import CodeCompletion
from codeverdict.evaluation.triage_engine import CodeVerdictTriageEngine
from codeverdict.evaluation.auto_evaluator import CodeVerdictAutoEvaluator
from codeverdict.evaluation.manual_evaluator import CodeVerdictManualEvaluator
from codeverdict.models.registry import CodeVerdictRegistry
import mlflow

async def evaluation_pipeline(eval_set_name: str, model_id: str, manual_dataset_name: str):
    """Main CodeVerdict evaluation pipeline"""
    print(f"🚀 Starting CodeVerdict evaluation for model: {model_id}")
    
    # Initialize components
    triage_engine = CodeVerdictTriageEngine()
    auto_evaluator = CodeVerdictAutoEvaluator()
    manual_evaluator = CodeVerdictManualEvaluator()
    registry = CodeVerdictRegistry(mlflow.get_tracking_uri())
    
    # Step 1: Generate or load completions
    completions = await _generate_completions(eval_set_name, model_id)
    print(f"📝 Generated {len(completions)} completions for evaluation")
    
    # Step 2: Triage completions
    auto_batch, manual_batch = triage_engine.triage_completions(completions)
    
    # Step 3: Auto-evaluate batch
    auto_results = []
    for completion in auto_batch:
        quality_scores = auto_evaluator.evaluate_code_quality(completion)
        verdict = auto_evaluator.generate_verdict(completion, quality_scores)
        auto_results.append(verdict)
        
        # Register verdict
        registry.register_verdict(verdict, model_id, completion.prompt_id)
    
    # Step 4: Setup manual evaluation
    manual_dataset = manual_evaluator.setup_code_quality_dataset(manual_dataset_name)
    manual_evaluator.add_completions_for_review(manual_dataset, manual_batch)
    manual_evaluator.push_to_argilla(manual_dataset, manual_dataset_name)
    
    # Step 5: Log summary
    with mlflow.start_run(run_name=f"eval_summary_{model_id}"):
        mlflow.log_params({
            "model_id": model_id,
            "eval_set": eval_set_name,
            "total_completions": len(completions),
            "auto_evaluated": len(auto_batch),
            "manual_review": len(manual_batch)
        })
        mlflow.log_metrics({
            "auto_approval_rate": len([r for r in auto_results if r["status"] == "auto_approved"]) / len(auto_results) if auto_results else 0,
            "average_auto_score": sum(r["overall_score"] for r in auto_results) / len(auto_results) if auto_results else 0
        })
    
    print(f"✅ CodeVerdict evaluation complete! Auto: {len(auto_results)}, Manual: {len(manual_batch)}")
    return {
        "auto_results": auto_results,
        "manual_tasks": len(manual_batch),
        "dashboard_url": f"http://localhost:6900/datasets/codeverdict/{manual_dataset_name}/"
    }

async def _generate_completions(eval_set_name: str, model_id: str) -> List[CodeCompletion]:
    """Generate completions for evaluation (placeholder implementation)"""
    # In a real implementation, this would call your model API
    # For now, we'll return sample completions
    
    sample_prompts = [
        {
            "id": "prompt_1",
            "text": "Write a function to reverse a string",
            "type": "code_generation",
            "difficulty": "easy"
        },
        {
            "id": "prompt_2", 
            "text": "Write a function to check if a number is prime",
            "type": "code_generation",
            "difficulty": "medium"
        },
        {
            "id": "prompt_3",
            "text": "Find the security vulnerability in this code: [code snippet]",
            "type": "security_audit", 
            "difficulty": "hard"
        }
    ]
    
    completions = []
    for prompt in sample_prompts:
        completion = CodeCompletion(
            prompt_id=prompt["id"],
            model_id=model_id,
            completion=_generate_sample_completion(prompt["text"]),
            metadata={
                "prompt_text": prompt["text"],
                "prompt_type": prompt["type"],
                "difficulty": prompt["difficulty"]
            }
        )
        completions.append(completion)
    
    return completions

def _generate_sample_completion(prompt: str) -> str:
    """Generate sample completion for demonstration"""
    if "reverse" in prompt.lower():
        return """
def reverse_string(s):
    return s[::-1]
"""
    elif "prime" in prompt.lower():
        return """
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
"""
    else:
        return """
# Security analysis: The code uses eval() which is dangerous
# Recommendation: Replace eval() with ast.literal_eval()
# or implement proper input validation
"""