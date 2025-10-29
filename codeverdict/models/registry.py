# codeverdict/models/registry.py
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from typing import List, Dict, Any, Optional
import json
import tempfile
import os
from datetime import datetime

from codeverdict.data.models import CodePrompt

class CodeVerdictRegistry:
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        
    def register_prompt_template(self, name: str, template: str, prompt_type: str, version: str = "1.0.0"):
        """Register a prompt template as an MLflow artifact"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_info = {
                "name": name,
                "template": template,
                "type": prompt_type,
                "version": version,
                "registered_at": datetime.now().isoformat()
            }
            
            prompt_path = os.path.join(tmp_dir, "prompt.json")
            with open(prompt_path, 'w') as f:
                json.dump(prompt_info, f)
            
            with mlflow.start_run(run_name=f"prompt_{name}"):
                mlflow.log_artifact(prompt_path, "prompts")
                mlflow.set_tag("codeverdict.prompt_name", name)
                mlflow.set_tag("codeverdict.prompt_type", prompt_type)
                mlflow.set_tag("codeverdict.version", version)
                
    def register_evaluation_set(self, name: str, prompts: List[CodePrompt], description: str = ""):
        """Register an evaluation set in MLflow"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            eval_set = {
                "name": name,
                "description": description,
                "prompts": [prompt.dict() for prompt in prompts],
                "created_at": datetime.now().isoformat(),
                "size": len(prompts)
            }
            
            eval_path = os.path.join(tmp_dir, "evaluation_set.json")
            with open(eval_path, 'w') as f:
                json.dump(eval_set, f)
            
            with mlflow.start_run(run_name=f"eval_set_{name}"):
                mlflow.log_artifact(eval_path, "evaluation_sets")
                mlflow.set_tag("codeverdict.eval_set_name", name)
                mlflow.set_tag("codeverdict.eval_set_size", len(prompts))
                mlflow.set_tag("codeverdict.type", "evaluation_set")
                
    def register_verdict(self, verdict: Dict[str, Any], model_id: str, prompt_id: str):
        """Register a final verdict in MLflow"""
        with mlflow.start_run(run_name=f"verdict_{verdict['id']}"):
            mlflow.log_params({
                "model_id": model_id,
                "prompt_id": prompt_id,
                "verdict_status": verdict["status"],
                "overall_score": verdict["overall_score"]
            })
            mlflow.log_metrics(verdict.get("criteria_scores", {}))
            mlflow.set_tag("codeverdict.verdict_id", verdict["id"])
            mlflow.set_tag("codeverdict.type", "verdict")