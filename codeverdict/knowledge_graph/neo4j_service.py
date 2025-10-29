# codeverdict/knowledge_graph/neo4j_service.py
from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from codeverdict.data.models import CodeCompletion, CodeVerdict, EvaluationResult

class CodeVerdictKnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
    
    def create_evaluation_run(self, run_id: str, model_id: str, timestamp: str):
        """Create evaluation run node in knowledge graph"""
        query = """
        CREATE (run:EvaluationRun {
            id: $run_id,
            model_id: $model_id,
            timestamp: $timestamp,
            created_at: datetime()
        })
        RETURN run
        """
        with self.driver.session() as session:
            session.run(query, run_id=run_id, model_id=model_id, timestamp=timestamp)
    
    def ingest_verdict_patterns(self, verdicts: List[CodeVerdict], run_id: str):
        """Ingest verdict patterns and create relationships"""
        query = """
        MATCH (run:EvaluationRun {id: $run_id})
        UNWIND $verdicts AS verdict
        CREATE (v:Verdict {
            id: verdict.id,
            status: verdict.status,
            overall_score: verdict.overall_score,
            confidence: verdict.confidence,
            completion_id: verdict.completion_id
        })
        CREATE (run)-[r:CONTAINS_VERDICT]->(v)
        
        // Create failure pattern clusters
        WITH v, verdict
        UNWIND keys(verdict.criteria_scores) AS criterion
        CREATE (c:Criterion {
            name: criterion,
            score: verdict.criteria_scores[criterion]
        })
        CREATE (v)-[sc:HAS_CRITERION]->(c)
        
        // Link similar failure patterns
        WITH v, verdict
        MATCH (existing:Verdict {status: verdict.status})
        WHERE existing.overall_score < 0.7 AND v.overall_score < 0.7
          AND abs(existing.overall_score - v.overall_score) < 0.1
        CREATE (existing)-[sim:SIMILAR_FAILURE]->(v)
        SET sim.strength = 1 - abs(existing.overall_score - v.overall_score)
        
        RETURN COUNT(v) as verdicts_ingested
        """
        
        verdict_dicts = [v.dict() for v in verdicts]
        with self.driver.session() as session:
            result = session.run(query, run_id=run_id, verdicts=verdict_dicts)
            return result.single()["verdicts_ingested"]

    def get_failure_clusters(self, model_id: str, threshold: float = 0.7):
        """Get clusters of similar failures for a model"""
        query = """
        MATCH (run:EvaluationRun {model_id: $model_id})-[:CONTAINS_VERDICT]->(v:Verdict)
        WHERE v.overall_score < $threshold
        MATCH (v)-[:HAS_CRITERION]->(c:Criterion)
        WITH v, COLLECT(c) AS criteria
        RETURN v.status as failure_type,
               COUNT(v) as frequency,
               AVG(v.overall_score) as avg_severity,
               COLLECT(criteria[0].name) as common_criteria
        ORDER BY frequency DESC
        """
        
        with self.driver.session() as session:
            result = session.run(query, model_id=model_id, threshold=threshold)
            return [dict(record) for record in result]