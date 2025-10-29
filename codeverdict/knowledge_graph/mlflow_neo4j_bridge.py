# codeverdict/knowledge_graph/mlflow_neo4j_bridge.py
import mlflow
from typing import Dict, Any
from .neo4j_service import CodeVerdictKnowledgeGraph

class MLflowNeo4jBridge:
    def __init__(self, kg_service: CodeVerdictKnowledgeGraph):
        self.kg = kg_service
        
    def track_mlflow_run_to_kg(self, run_id: str):
        """Bridge MLflow run data to knowledge graph"""
        run = mlflow.get_run(run_id)
        
        # Extract key metrics and parameters
        metrics = run.data.metrics
        params = run.data.params
        tags = run.data.tags
        
        # Create evaluation run in knowledge graph
        self.kg.create_evaluation_run(
            run_id=run_id,
            model_id=params.get('model_id', 'unknown'),
            timestamp=run.info.start_time.isoformat()
        )
        
        # Link metrics and parameters
        self._link_mlflow_artifacts(run_id, metrics, params, tags)
        
    def _link_mlflow_artifacts(self, run_id: str, metrics: Dict, params: Dict, tags: Dict):
        """Link MLflow artifacts to knowledge graph nodes"""
        query = """
        MATCH (run:EvaluationRun {id: $run_id})
        
        // Create metric nodes
        UNWIND keys($metrics) AS metric_name
        CREATE (m:Metric {
            name: metric_name,
            value: $metrics[metric_name]
        })
        CREATE (run)-[rm:HAS_METRIC]->(m)
        
        // Create parameter nodes  
        UNWIND keys($params) AS param_name
        CREATE (p:Parameter {
            name: param_name,
            value: $params[param_name]
        })
        CREATE (run)-[rp:WITH_PARAMETER]->(p)
        
        // Create improvement trajectory
        WITH run, $metrics AS metrics
        MATCH (prev:EvaluationRun {model_id: run.model_id})
        WHERE prev.timestamp < run.timestamp
        WITH run, prev, metrics
        ORDER BY prev.timestamp DESC
        LIMIT 1
        
        CREATE (prev)-[imp:IMPROVES_TO]->(run)
        SET imp.delta_pass1 = metrics.pass1 - prev.metrics.pass1
        SET imp.improvement_areas = $improvement_areas
        
        RETURN run.id
        """
        
        improvement_areas = self._calculate_improvement_areas(metrics, params)
        
        with self.kg.driver.session() as session:
            session.run(query, 
                       run_id=run_id, 
                       metrics=metrics, 
                       params=params,
                       improvement_areas=improvement_areas)

    def _calculate_improvement_areas(self, metrics: Dict, params: Dict) -> List[str]:
        """Calculate key improvement areas from metrics"""
        improvement_areas = []
        
        if metrics.get('correctness', 1) < 0.7:
            improvement_areas.append("correctness")
        if metrics.get('security', 1) < 0.8:
            improvement_areas.append("security")
        if metrics.get('efficiency', 1) < 0.6:
            improvement_areas.append("efficiency")
            
        return improvement_areas