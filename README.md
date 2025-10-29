# CodeVerdict ⚖️: AI Code Evaluation & Improvement Platform

> **Where AI Code Stands Trial** - A comprehensive evaluation platform that transforms code quality assessment into intelligent hill climbing through knowledge graph-powered insights.

![CodeVerdict Architecture](https://img.shields.io/badge/Architecture-Production%20Ready-green)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-orange)
![Neo4j](https://img.shields.io/badge/Neo4j-5.0%2B-blue)

## 🚀 What is CodeVerdict?

CodeVerdict is an enterprise-grade AI code evaluation platform that goes beyond traditional metrics to provide **intelligent, actionable insights** for model improvement. We transform raw evaluation data into strategic improvement intelligence through:

- **🤖 Smart Triage**: 50/50 auto-manual evaluation split with LLM-as-judge
- **📊 Multi-Dimensional Metrics**: Pass@k, code quality, security, and beyond
- **🧠 Knowledge Graph Intelligence**: Neo4j-powered pattern discovery and hill climbing
- **🔬 Continuous Improvement**: Automated intervention recommendations
- **📈 Production Ready**: FastAPI, MLflow, and enterprise tooling

## 🏗️ System Architecture

```
CodeVerdict Core
├── 🎯 Evaluation Engine
│   ├── Auto Evaluator (LLM-as-Judge)
│   ├── Manual Evaluator (Argilla Integration)
│   └── Triage Engine (50/50 Smart Split)
├── 📚 Knowledge Graph  
│   ├── Neo4j Pattern Storage
│   ├── MLflow-Neo4j Bridge
│   └── AI Agent Query Service
├── 🔬 Model Registry
│   ├── MLflow Experiment Tracking
│   └── Enhanced Registry with KG Insights
└── 🌐 API Layer
    ├── FastAPI REST API
    └── Real-time Dashboard
```
<img width="777" height="1276" alt="image" src="https://github.com/user-attachments/assets/a17a1464-9a00-4e34-8f58-07d69f58762d" />


## 📁 Complete File Structure

```bash
codeverdict/
├── 📁 api/
│   └── main.py                          # FastAPI application & endpoints
├── 📁 config/
│   └── settings.py                      # Pydantic settings management
├── 📁 data/
│   └── models.py                        # Pydantic data models
├── 📁 evaluation/
│   ├── auto_evaluator.py               # LLM-as-judge auto evaluation
│   ├── manual_evaluator.py             # Argilla manual evaluation setup
│   └── triage_engine.py                # Smart 50/50 triage logic
├── 📁 knowledge_graph/
│   ├── neo4j_service.py                # Neo4j connection & operations
│   ├── mlflow_neo4j_bridge.py          # MLflow to Neo4j bridge
│   └── agent_queries.py                # AI agent query service
├── 📁 models/
│   ├── registry.py                     # MLflow model registry
│   └── enhanced_registry.py            # KG-enhanced registry
├── 📁 orchestration/
│   └── workflows.py                    # Evaluation pipeline workflows
├── 📁 utils/
│   └── helpers.py                      # Utility functions
├── 📁 tests/
│   ├── test_evaluation.py
│   ├── test_triage.py
│   └── test_knowledge_graph.py
├── requirements.txt
├── docker-compose.yml
├── .env.example
└── README.md
```

## 🛠️ Quick Start

### 1. Installation & Setup

```bash
# Clone the repository
git clone https://github.com/cklam12345/codeverdict.git
cd codeverdict

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys and configurations
```

### 2. Start Services with Docker

```bash
# Start all services (MLflow, Neo4j, Argilla, FastAPI)
docker-compose up -d
```

### 3. Initialize CodeVerdict

```python
from codeverdict.api.main import app
from codeverdict.config.settings import settings

# The system automatically initializes with sample evaluation sets
# Visit http://localhost:8000/docs to explore the API
```

## 📊 Core Evaluation Metrics

CodeVerdict evaluates AI-generated code across multiple dimensions:

### Functional Correctness
- **Pass@k**: Probability of correct solution in k attempts
- **G-Pass@k**: Generalization-focused Pass@k with hidden tests
- **Test Coverage**: Percentage of test cases passed
- **Edit Distance to Fix**: Characters needed to fix incorrect code

### Code Quality
- **Readability Score**: Code understandability (1-5)
- **Efficiency Score**: Algorithm performance (1-5)
- **Security Score**: Vulnerability detection (1-5)
- **Style Adherence**: Coding standards compliance (1-5)

### Business Impact
- **Time-to-Correct-Solution**: Developer productivity metric
- **Adoption Rate**: AI suggestion acceptance rate
- **Cost-per-Correct-Solution**: Economic efficiency

## 🧠 Knowledge Graph Intelligence

Our Neo4j knowledge graph enables intelligent hill climbing:

```python
from codeverdict.knowledge_graph.agent_queries import HillClimbingAgentQueries

# Get AI-powered improvement recommendations
agent = HillClimbingAgentQueries(kg_service)
interventions = agent.find_high_roi_interventions("model-v2")

# Returns:
# [
#   {
#     "failure_type": "recursion_base_cases",
#     "failure_count": 45,
#     "fixability_score": 0.89,
#     "expected_lift": "+12% Pass@1"
#   }
# ]
```

### Sample Graph Queries

```cypher
// Find similar failure patterns across models
MATCH (v:Verdict)-[sim:SIMILAR_FAILURE]-(other:Verdict)
WHERE v.overall_score < 0.7
RETURN v.status, COUNT(sim) as pattern_frequency
ORDER BY pattern_frequency DESC

// Get improvement trajectory
MATCH (model:Model)-[:EVALUATED_IN]->(run:EvaluationRun)
WITH run ORDER BY run.timestamp
RETURN run.timestamp, run.metrics.pass1
```

## 🎯 Smart Triage System

Our 50/50 auto-manual split intelligently routes evaluations:

```python
from codeverdict.evaluation.triage_engine import CodeVerdictTriageEngine

triage_engine = CodeVerdictTriageEngine(
    auto_eval_threshold=0.8,
    manual_sample_rate=0.5
)

auto_batch, manual_batch = triage_engine.triage_completions(completions)
print(f"🔀 Auto: {len(auto_batch)}, Manual: {len(manual_batch)}")
```

### Triage Logic:
- **Security Audits**: Always manual review
- **High-Quality Code**: Auto-approve (confidence > 0.8)
- **Borderline Cases**: Manual review + sampling
- **Critical Failures**: Auto-reject with detailed analysis

## 📈 MLflow Integration

Track every evaluation with comprehensive experiment tracking:

```python
from codeverdict.models.registry import CodeVerdictRegistry

registry = CodeVerdictRegistry(settings.mlflow_tracking_uri)

# Register evaluation results
registry.register_verdict(
    verdict=final_verdict,
    model_id="codegen-2b",
    prompt_id="fibonacci_recursive"
)
```

## 🔧 API Usage

### Start Evaluation
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "your-model-v1",
    "eval_set_name": "codeverdict_sample_evals"
  }'
```

### Get Results
```bash
curl "http://localhost:8000/results/your-model-v1"
```

### Knowledge Graph Insights
```bash
curl "http://localhost:8000/kg/insights/your-model-v1"
```

## 🚀 Advanced Features

### 1. Real-time Hill Climbing
```python
# Automated improvement recommendations
insights = agent_queries.find_high_roi_interventions("model-v2")
training_spec = insights.generate_training_curriculum()
```

### 2. Failure Pattern Clustering
```python
# Group similar failures for targeted fixes
clusters = agent_queries.find_similar_failure_clusters([
    "recursion_errors", "off_by_one"
])
```

### 3. Intervention ROI Prediction
```python
# Predict improvement before training
predicted_lift = kg_service.predict_intervention_roi(
    intervention="synthetic_tree_data",
    historical_success_rate=0.85
)
```

## 🏭 Production Deployment

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  fastapi:
    build: .
    ports: ["8000:8000"]
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - MLFLOW_TRACKING_URI=http://mlflow:5000
  
  mlflow:
    image: mlflow/mlflow:latest
    ports: ["5000:5000"]
  
  neo4j:
    image: neo4j:5.0
    ports: ["7687:7687", "7474:7474"]
  
  argilla:
    image: argilla/argilla-server:latest
    ports: ["6900:6900"]
```

### Environment Variables
```bash
# .env
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=codeverdict
ARGILLA_API_URL=http://localhost:6900
OPENAI_API_KEY=your-key-here
```

## 📊 Sample Evaluation Results

After running CodeVerdict, you get:

```json
{
  "model_id": "codegen-2b",
  "summary": {
    "total_prompts": 150,
    "auto_evaluated": 75,
    "manual_reviewed": 75,
    "average_score": 0.82,
    "verdict_distribution": {
      "auto_approved": 60,
      "manual_approved": 68, 
      "rejected": 22
    }
  },
  "improvement_insights": {
    "high_roi_interventions": [
      {
        "target": "recursion_base_cases",
        "expected_lift": "+12%",
        "effort_required": "low",
        "historical_success_rate": 0.89
      }
    ],
    "predicted_next_score": 0.89
  }
}
```

## 🔬 Research Integration

CodeVerdict accelerates research cycles:

### Before CodeVerdict
```
Evaluation → Aggregate Scores → Guess Improvements → Train Blindly (4-6 weeks)
```

### After CodeVerdict  
```
Evaluation → Pattern Analysis → Targeted Interventions → Measured Improvement (1-2 weeks)
                                      ↑
                             Knowledge Graph Wisdom
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **MLflow** for experiment tracking
- **Neo4j** for knowledge graph capabilities  
- **Argilla** for human-in-the-loop evaluation
- **FastAPI** for high-performance API framework
- **Phoenix** for ML observability

---

## 🚀 Ready to Transform AI Evaluation?

CodeVerdict turns evaluation data into improvement intelligence. Stop guessing what to fix next - let the knowledge graph guide your hill climbing.

**Get started:**
```bash
docker-compose up -d
curl -X POST "http://localhost:8000/evaluate" -H "Content-Type: application/json" -d '{"model_id": "your-model"}'
```

Visit:
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000  
- **Argilla**: http://localhost:6900
- **Neo4j Browser**: http://localhost:7474

**Transform your AI evaluation from metrics to intelligence!** 🚀
