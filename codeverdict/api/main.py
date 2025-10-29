# codeverdict/api/main.py
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import mlflow

from codeverdict.config.settings import settings
from codeverdict.orchestration.workflows import evaluation_pipeline
from codeverdict.data.models import CodePrompt
from codeverdict.models.registry import CodeVerdictRegistry

# Initialize FastAPI app with CodeVerdict branding
app = FastAPI(
    title="CodeVerdict",
    description="⚖️ Where AI Code Stands Trial - Comprehensive AI Code Evaluation Platform",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_registry = CodeVerdictRegistry(settings.mlflow_tracking_uri)

class EvaluationRequest(BaseModel):
    model_id: str
    eval_set_name: str = "default_code_evals"
    manual_dataset_name: Optional[str] = None

class EvaluationResponse(BaseModel):
    evaluation_id: str
    status: str
    auto_results: int
    manual_tasks: int
    dashboard_url: str

@app.on_event("startup")
async def startup_event():
    """Initialize CodeVerdict services on startup"""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    
    # Create sample evaluation set for demonstration
    try:
        sample_prompts = [
            CodePrompt(
                prompt="Write a Python function to calculate fibonacci numbers recursively",
                prompt_type="code_generation",
                difficulty="easy",
                tags=["algorithm", "recursion"]
            ),
            CodePrompt(
                prompt="Explain the time complexity of the fibonacci function and suggest optimizations",
                prompt_type="code_explanation", 
                difficulty="medium",
                tags=["complexity", "optimization"]
            ),
            CodePrompt(
                prompt="Find and fix the security vulnerability in this authentication function: [vulnerable code]",
                prompt_type="security_audit",
                difficulty="hard",
                tags=["security", "authentication"]
            )
        ]
        
        model_registry.register_evaluation_set(
            name="codeverdict_sample_evals",
            prompts=sample_prompts,
            description="Sample evaluation set for CodeVerdict demonstration"
        )
        print("✅ CodeVerdict sample evaluation set created")
    except Exception as e:
        print(f"ℹ️ Sample evaluation set may already exist: {e}")

@app.get("/")
async def root():
    """CodeVerdict API root endpoint"""
    return {
        "message": "Welcome to CodeVerdict ⚖️",
        "description": "Where AI Code Stands Trial",
        "version": settings.app_version,
        "endpoints": {
            "evaluate": "/evaluate/{model_id}",
            "results": "/results/{model_id}",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    services = {
        "mlflow": "healthy",
        "argilla": "healthy", 
        "evaluation_engine": "healthy"
    }
    return {"status": "healthy", "services": services}

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_model(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Trigger CodeVerdict evaluation pipeline for a model"""
    try:
        manual_dataset = request.manual_dataset_name or f"codeverdict_{request.model_id}_{request.eval_set_name}"
        
        # Run evaluation in background
        background_tasks.add_task(
            evaluation_pipeline,
            eval_set_name=request.eval_set_name,
            model_id=request.model_id,
            manual_dataset_name=manual_dataset
        )
        
        return EvaluationResponse(
            evaluation_id=f"eval_{request.model_id}_{request.eval_set_name}",
            status="started",
            auto_results=0,  # Would be updated from actual run
            manual_tasks=0,  # Would be updated from actual run
            dashboard_url=f"http://localhost:6900/datasets/codeverdict/{manual_dataset}/"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CodeVerdict evaluation failed: {str(e)}")

@app.get("/results/{model_id}")
async def get_evaluation_results(model_id: str):
    """Get CodeVerdict evaluation results for a model"""
    # This would query your results database
    return {
        "model_id": model_id,
        "status": "evaluation_complete",
        "summary": {
            "total_prompts": 50,
            "auto_evaluated": 25,
            "manual_reviewed": 25,
            "average_score": 0.82,
            "verdict_distribution": {
                "auto_approved": 20,
                "manual_approved": 23,
                "rejected": 7
            }
        },
        "dashboard": {
            "mlflow": "http://localhost:5000",
            "argilla": "http://localhost:6900"
        }
    }

@app.get("/prompts")
async def list_evaluation_sets():
    """List available evaluation sets in CodeVerdict"""
    return {
        "evaluation_sets": [
            {
                "name": "codeverdict_sample_evals",
                "description": "Sample evaluation set for CodeVerdict",
                "size": 3,
                "types": ["code_generation", "code_explanation", "security_audit"]
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "codeverdict.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None
    )