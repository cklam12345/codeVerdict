from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import uuid

class VerdictStatus(str, Enum):
    PENDING = "pending"
    AUTO_APPROVED = "auto_approved"
    MANUAL_REVIEW = "manual_review"
    REJECTED = "rejected"
    APPROVED = "approved"

class CodeVerdict(BaseModel):
    """Final verdict for a code completion"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    completion_id: str
    status: VerdictStatus
    overall_score: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    criteria_scores: Dict[str, float] = Field(default_factory=dict)
    human_feedback: Optional[str] = None
    reviewed_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class CodeEvaluationCriteria(BaseModel):
    correctness: float = Field(..., ge=0, le=1)
    efficiency: float = Field(..., ge=0, le=1) 
    readability: float = Field(..., ge=0, le=1)
    security: float = Field(..., ge=0, le=1)
    style_adherence: float = Field(..., ge=0, le=1)

class CodePrompt(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    prompt_type: str
    context: Optional[Dict[str, Any]] = None
    expected_output: Optional[str] = None
    test_cases: Optional[List[Dict]] = None
    difficulty: str = "medium"
    tags: List[str] = Field(default_factory=list)

class CodeCompletion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt_id: str
    model_id: str
    completion: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class EvaluationResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    completion_id: str
    evaluator_type: str
    criteria: CodeEvaluationCriteria
    overall_score: float
    confidence: Optional[float] = None
    rater_id: Optional[str] = None
    feedback: Optional[str] = None
    verdict: Optional[VerdictStatus] = None