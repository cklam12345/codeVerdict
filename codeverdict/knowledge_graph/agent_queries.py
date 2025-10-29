# codeverdict/knowledge_graph/agent_queries.py
from .neo4j_service import CodeVerdictKnowledgeGraph
from typing import List, Dict, Any

class HillClimbingAgentQueries:
    def __init__(self, kg_service: CodeVerdictKnowledgeGraph):
        self.kg = kg_service
        
    def find_high_roi_interventions(self, model_id: str, limit: int = 5):
        """Find highest ROI interventions based on historical patterns"""
        query = """
        // Find failure patterns that are prevalent but easily fixable
        MATCH (run:EvaluationRun {model_id: $model_id})-[:CONTAINS_VERDICT]->(v:Verdict)
        WHERE v.overall_score < 0.7
        MATCH (v)-[:HAS_CRITERION]->(c:Criterion)
        
        WITH c.name AS failure_type, 
             COUNT(v) as failure_count,
             AVG(c.score) as avg_severity,
             COUNT(DISTINCT run) as run_prevalence
        
        // Calculate fixability (inverse of severity and prevalence)
        WITH failure_type, failure_count, avg_severity, run_prevalence,
             (1 - avg_severity) * failure_count as fixability_score
        
        ORDER BY fixability_score DESC
        LIMIT $limit
        
        RETURN failure_type, failure_count, avg_severity, fixability_score
        """
        
        with self.kg.driver.session() as session:
            result = session.run(query, model_id=model_id, limit=limit)
            return [dict(record) for record in result]
    
    def get_improvement_trajectory(self, model_id: str):
        """Get the improvement trajectory for a model"""
        query = """
        MATCH (run:EvaluationRun {model_id: $model_id})
        WITH run
        ORDER BY run.timestamp ASC
        
        RETURN {
            run_id: run.id,
            timestamp: run.timestamp,
            model_id: run.model_id
        } AS trajectory_point
        ORDER BY run.timestamp
        """
        
        with self.kg.driver.session() as session:
            result = session.run(query, model_id=model_id)
            return [record["trajectory_point"] for record in result]
    
    def find_similar_failure_clusters(self, target_failures: List[str]):
        """Find similar failure clusters across models"""
        query = """
        MATCH (v:Verdict)-[:HAS_CRITERION]->(c:Criterion)
        WHERE c.name IN $target_failures AND c.score < 0.6
        
        WITH v, COLLECT(c.name) AS failure_patterns
        MATCH (v)-[:SIMILAR_FAILURE*1..2]-(similar:Verdict)
        
        WITH similar, failure_patterns,
             COUNT(v) AS cluster_size,
             AVG(similar.overall_score) AS cluster_avg_score
        
        WHERE cluster_size > 3  // Significant clusters only
        RETURN similar.status AS cluster_type,
               cluster_size,
               cluster_avg_score,
               COLLECT(DISTINCT similar.completion_id)[0..5] AS example_failures,
               failure_patterns
        ORDER BY cluster_size DESC
        """
        
        with self.kg.driver.session() as session:
            result = session.run(query, target_failures=target_failures)
            return [dict(record) for record in result]