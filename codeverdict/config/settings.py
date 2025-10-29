from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
from enum import Enum

class EvaluationType(str, Enum):
    CODE_QUALITY = "code_quality"
    SAFETY = "safety" 
    CORRECTNESS = "correctness"
    EFFICIENCY = "efficiency"
    STYLE = "style"

class PromptType(str, Enum):
    CODE_GENERATION = "code_generation"
    CODE_EXPLANATION = "code_explanation"
    BUG_FIXING = "bug_fixing"
    CODE_REVIEW = "code_review"
    SECURITY_AUDIT = "security_audit"

class Settings(BaseSettings):
    # CodeVerdict Configuration
    app_name: str = "CodeVerdict"
    app_version: str = "1.0.0"
    description: str = "Where AI Code Stands Trial"
    
    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_registry_uri: str = "sqlite:///mlflow.db"
    
    # Argilla
    argilla_api_url: str = "http://localhost:6900"
    argilla_api_key: str = "codeverdict.apikey"
    
    # Neo4j Knowledge Graph
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "codeverdict"
    
    # Model APIs
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    
    # Evaluation
    auto_eval_threshold: float = 0.8
    manual_review_sample_rate: float = 0.5
    min_manual_reviews: int = 10
    max_concurrent_evaluations: int = 50
    
    # Security
    enable_security_scan: bool = True
    enable_bias_detection: bool = True
    
    # Knowledge Graph Settings
    enable_knowledge_graph: bool = True
    kg_auto_ingest: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
