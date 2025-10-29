# codeverdict/evaluation/manual_evaluator.py
import argilla as rg
from typing import List, Dict, Any, Optional
from codeverdict.data.models import CodeCompletion, CodeEvaluationCriteria, EvaluationResult
import pandas as pd
import numpy as np

class CodeVerdictManualEvaluator:
    def __init__(self, api_url: str = "http://localhost:6900", api_key: str = "codeverdict.apikey"):
        rg.init(api_url=api_url, api_key=api_key)
        
    def create_evaluation_workspace(self, workspace_name: str, guidelines: str):
        """Create a workspace for manual evaluation in CodeVerdict"""
        try:
            rg.Workspace.create(workspace_name)
            print(f"✅ Created CodeVerdict workspace: {workspace_name}")
        except Exception as e:
            print(f"ℹ️ Workspace may already exist: {e}")
            
    def setup_code_quality_dataset(self, dataset_name: str, workspace: str = "codeverdict"):
        """Setup dataset for code quality evaluation"""
        settings = rg.DatasetSettings(
            guidelines="""
            # CodeVerdict Evaluation Guidelines ⚖️
            
            Evaluate the AI-generated code based on the following criteria:
            
            ## Correctness (1-5)
            - Does the code solve the problem correctly?
            - Are there any logical errors?
            - Does it handle edge cases?
            
            ## Efficiency (1-5)  
            - Is the code optimized for performance?
            - Are there unnecessary computations?
            - Is the algorithm choice appropriate?
            
            ## Readability (1-5)
            - Is the code easy to understand?
            - Are variables and functions well-named?
            - Is the code properly structured?
            
            ## Security (1-5)
            - Does it follow security best practices?
            - Are there potential vulnerabilities?
            - Is input validation performed?
            
            ## Style Adherence (1-5)
            - Does it follow proper coding style?
            - Is the formatting consistent?
            - Are comments used appropriately?
            """,
            fields=[
                rg.TextField(name="prompt", required=True),
                rg.TextField(name="completion", required=True),
                rg.TextField(name="model_id", required=True)
            ],
            questions=[
                rg.RatingQuestion(
                    name="correctness",
                    title="How correct is the code?",
                    description="Does it solve the problem correctly without errors?",
                    required=True,
                    values=[1, 2, 3, 4, 5]
                ),
                rg.RatingQuestion(
                    name="efficiency", 
                    title="How efficient is the code?",
                    description="Is it optimized for performance?",
                    required=True,
                    values=[1, 2, 3, 4, 5]
                ),
                rg.RatingQuestion(
                    name="readability",
                    title="How readable is the code?",
                    description="Is it easy to understand and maintain?",
                    required=True,
                    values=[1, 2, 3, 4, 5]
                ),
                rg.RatingQuestion(
                    name="security",
                    title="How secure is the code?",
                    description="Does it follow security best practices?",
                    required=True,
                    values=[1, 2, 3, 4, 5]
                ),
                rg.RatingQuestion(
                    name="style_adherence",
                    title="How well does it adhere to coding style?",
                    description="Does it follow proper style guidelines?",
                    required=True,
                    values=[1, 2, 3, 4, 5]
                ),
                rg.TextQuestion(
                    name="feedback",
                    title="Additional feedback",
                    description="Any specific issues, suggestions, or comments?",
                    required=False
                )
            ]
        )
        
        dataset = rg.FeedbackDataset(
            settings=settings,
            fields=settings.fields,
            questions=settings.questions
        )
        
        return dataset
    
    def add_completions_for_review(self, dataset: rg.FeedbackDataset, completions: List[CodeCompletion]):
        """Add completions to the CodeVerdict manual evaluation queue"""
        records = []
        
        for completion in completions:
            record = rg.FeedbackRecord(
                fields={
                    "prompt": completion.metadata.get("prompt_text", ""),
                    "completion": completion.completion,
                    "model_id": completion.model_id
                },
                metadata={
                    "completion_id": completion.id,
                    "prompt_id": completion.prompt_id,
                    "prompt_type": completion.metadata.get("prompt_type", "unknown"),
                    "difficulty": completion.metadata.get("difficulty", "medium"),
                    "timestamp": completion.timestamp.isoformat()
                }
            )
            records.append(record)
            
        dataset.add_records(records)
        print(f"📝 Added {len(records)} completions to manual review queue")
        
    def push_to_argilla(self, dataset: rg.FeedbackDataset, dataset_name: str):
        """Push dataset to Argilla server with CodeVerdict branding"""
        dataset.push_to_argilla(name=dataset_name, workspace="codeverdict")
        print(f"🚀 Published CodeVerdict dataset: {dataset_name}")