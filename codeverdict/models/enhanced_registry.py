# codeverdict/models/enhanced_registry.py
from .registry import CodeVerdictRegistry
from codeverdict.knowledge_graph.mlflow_neo4j_bridge import MLflowNeo4jBridge
from codeverdict.knowledge_graph.neo4j_service import CodeVerdictKnowledgeGraph
import mlflow
from typing import Dict, Any

class EnhancedCodeVerdictRegistry(CodeVerdictRegistry):
    def __init__(self, tracking_uri: str, neo4j_uri: str, neo4j_auth: tuple):
        super().__init__(tracking_uri)
        self.kg_service = CodeVerdictKnowledgeGraph(neo4j_uri, neo4j_auth[0], neo4j_auth[1])
        self.kg_bridge = MLflowNeo4jBridge(self.kg_service)
    
    def register_verdict_with_insights(self, verdict: Dict[str, Any], model_id: str, prompt_id: str):
        """Enhanced verdict registration with knowledge graph insights"""
        # Original MLflow registration
        super().register_verdict(verdict, model_id, prompt_id)
        
        # Get the current run ID
        current_run = mlflow.active_run()
        if current_run:
            # Bridge to knowledge graph
            self.kg_bridge.track_mlflow_run_to_kg(current_run.info.run_id)
            
            # Generate real-time insights
            insights = self._generate_real_time_insights(verdict, model_id)
            
            # Log insights back to MLflow
            mlflow.log_metrics(insights.get('metrics', {}))
            mlflow.log_params(insights.get('params', {}))
    
    def _generate_real_time_insights(self, verdict: Dict, model_id: str) -> Dict:
        """Generate real-time insights using knowledge graph"""
        from codeverdict.knowledge_graph.agent_queries import HillClimbingAgentQueries
        agent_queries = HillClimbingAgentQueries(self.kg_service)
        
        # Find similar historical patterns
        if verdict['overall_score'] < 0.7:
            failure_types = list(verdict.get('criteria_scores', {}).keys())
            similar_clusters = agent_queries.find_similar_failure_clusters(failure_types)
            
            return {
                'metrics': {
                    'similar_cluster_size': len(similar_clusters) if similar_clusters else 0,
                },
                'params': {
                    'failure_pattern_match': similar_clusters[0]['cluster_type'] if similar_clusters else 'new_pattern'
                }
            }
        return {}