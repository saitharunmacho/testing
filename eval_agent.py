import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import requests
from databricks.sdk import WorkspaceClient
import re
import mlflow
from fastapi import FastAPI, HTTPException
import uvicorn
import openpyxl
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from SQL evaluation"""
    rating: str = "Bad"
    explanation: str = ""
    score: float = 0.0
    comparison_details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.comparison_details is None:
            self.comparison_details = {}

class DatabricksClient:
    """Handles Databricks API calls"""
    
    def __init__(self, workspace_url: str, token: str):
        self.workspace_url = workspace_url.rstrip('/')
        self.token = token 
        self.client = WorkspaceClient(host=self.workspace_url, token=self.token)
    
    def call_llm(self, prompt: str, model: str = "databricks-gpt-oss-120b") -> str:
        try:
            client = OpenAI(
                api_key=self.token,
                base_url=f"{self.workspace_url}/serving-endpoints",
            )
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model,
                max_tokens=1000,
                temperature=0.1
            )
            print(f"----------response: {chat_completion}")
            
            response_content = chat_completion.choices[0].message.content
            return response_content 
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return '{"rating": "Bad", "score": 0.0, "explanation": "LLM error"}'

class SQLComparator:
    """Handles SQL comparison following Genie rules using LLM"""
    
    def __init__(self, databricks_client):
        self.databricks = databricks_client
    
    @staticmethod
    def normalize_sql(sql: str) -> str:
        normalized = re.sub(r'\s+', ' ', sql.strip())
        normalized = re.sub(r';\s*$', '', normalized)
        return normalized.upper()
    
    @staticmethod
    def compare_exact_sql(query1: str, query2: str) -> bool:
        return SQLComparator.normalize_sql(query1) == SQLComparator.normalize_sql(query2)
    
    def evaluate_sql_comparison(self, generated_sql: str, benchmark_sql: str) -> Dict[str, Any]:
        try:
            if self.compare_exact_sql(generated_sql, benchmark_sql):
                return {
                    'rating': 'Good',
                    'score': 1.0,
                    'explanation': 'SQL exactly matches the provided SQL Answer',
                    'rule_applied': 'Exact SQL Match',
                    'details': {
                        'normalized_generated': self.normalize_sql(generated_sql),
                        'normalized_benchmark': self.normalize_sql(benchmark_sql)
                    }
                }
            
            return self._llm_sql_evaluation(generated_sql, benchmark_sql)
            
        except Exception as e:
            logger.error(f"Error in SQL comparison: {e}")
            return {
                'rating': 'Bad',
                'score': 0.0,
                'explanation': f'Error during SQL comparison: {str(e)}',
                'rule_applied': 'Error Handling',
                'details': {'error': str(e)}
            }
    
    def _llm_sql_evaluation(self, generated_sql: str, benchmark_sql: str) -> Dict[str, Any]:
        try:
            prompt = f"""
                    Evaluate these two SQL queries for Databricks SQL compatibility and equivalence:

                    Generated SQL: {generated_sql}
                    Benchmark SQL: {benchmark_sql}

                    Apply these Genie rules:
                    1. Good: SQL exactly matches the provided SQL Answer
                    2. Good: Result set exactly matches the SQL Answer result set  
                    3. Good: Result set has same data but sorted differently
                    4. Good: Numeric values round to same 4 significant digits
                    5. Bad: SQL produces empty result set or returns error
                    6. Bad: Result set includes extra columns compared to SQL Answer
                    7. Bad: Single cell result is different from SQL Answer

                    Check if the generated SQL is compatible with Databricks SQL syntax.
                    Check if the queries would produce equivalent results.

                    Respond with JSON:
                    {{
                        "rating": "Good" or "Bad",
                        "score": 0.0 to 1.0,
                        "explanation": "Detailed explanation",
                        "rule_applied": "Which rule was applied",
                        "databricks_compatible": true or false,
                        "details": {{
                            "syntax_check": "SQL syntax analysis",
                            "equivalence_check": "Result equivalence analysis"
                        }}
                    }}
                    """
            
            response = self.databricks.call_llm(prompt)
            return self._parse_llm_response(response)
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return {
                'rating': 'Bad',
                'score': 0.0,
                'explanation': f'LLM evaluation failed: {str(e)}',
                'rule_applied': 'LLM Error',
                'details': {'error': str(e)}
            }
    
    def _parse_llm_response(self, response) -> Dict[str, Any]:
        try:
            print(f"----------parse_llm_response: {response}")
            
            response_text = ""
            if isinstance(response, list):
                for item in response:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        response_text = item.get('text', '')
                        break
            elif isinstance(response, str):
                response_text = response
            else:
                logger.warning(f"Unexpected response type: {type(response)}")
                response_text = str(response)
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    'rating': result.get('rating', 'Bad'),
                    'score': result.get('score', 0.0),
                    'explanation': result.get('explanation', 'No explanation provided'),
                    'rule_applied': result.get('rule_applied', 'LLM Analysis'),
                    'details': result.get('details', {})
                }
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
        
        return {
            'rating': 'Bad',
            'score': 0.0,
            'explanation': 'Could not parse LLM response',
            'rule_applied': 'Parse Error',
            'details': {'parse_error': True}
        }

class SQLEvaluationAgent:
    """Main SQL evaluation agent following Genie rules"""
    
    def __init__(
        self,
        workspace_url: str,
        token: str,
        benchmark_file_path: str,
        model_name: str = ""
    ):
        self.workspace_url = workspace_url.rstrip('/')
        self.token = token
        self.model_name = model_name
        
        self.databricks = DatabricksClient(workspace_url, token)
        
        self.sql_comparator = SQLComparator(self.databricks)
        
        self.benchmark_data = self._load_benchmark_data(benchmark_file_path)
        
        logger.info(f"Agent initialized with {len(self.benchmark_data)} benchmark queries")
    
    def _load_benchmark_data(self, file_path: str) -> pd.DataFrame:
        try:
            if file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.xls'):
                data = pd.read_excel(file_path)
            else:
                data = pd.read_csv(file_path)
            
            required_columns = ['User Query', 'SQL Query', 'Classification']
            missing = [col for col in required_columns if col not in data.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            
            data = data.dropna(subset=required_columns)
            data['User Query'] = data['User Query'].str.strip()
            data['SQL Query'] = data['SQL Query'].str.strip()
            return data
            
        except Exception as e:
            logger.error(f"Error loading benchmark data: {e}")
            raise
    
    def evaluate(
        self,
        user_query: str,
        generated_sql: str
    ) -> EvaluationResult:

        try:
            best_match = self._find_best_match(user_query)
            if best_match is None:
                return EvaluationResult(
                    rating="Bad",
                    explanation="No matching benchmark query found",
                    score=0.0,
                    comparison_details={'error': 'No benchmark found'}
                )
            benchmark_sql = best_match['SQL Query']
            match_info = best_match
            
            comparison = self.sql_comparator.evaluate_sql_comparison(generated_sql, benchmark_sql)
            
            return EvaluationResult(
                rating=comparison['rating'],
                explanation=comparison['explanation'],
                score=comparison['score'],
                comparison_details={
                    'rule_applied': comparison['rule_applied'],
                    'details': comparison['details'],
                    'match_info': match_info
                }
            )
            
        except Exception as e:  
            logger.error(f"Evaluation error: {e}")
            return EvaluationResult(
                rating="Bad",
                explanation=f"Evaluation failed: {e}",
                score=0.0,
                comparison_details={'error': str(e)}
            )
    
    def _find_best_match(self, user_query: str) -> Optional[Dict[str, Any]]:
        try:
            best_match = None
            best_similarity = 0.0
            
            for _, row in self.benchmark_data.iterrows():
                query_words = set(user_query.lower().split())
                benchmark_words = set(row['User Query'].lower().split())
                
                if query_words and benchmark_words:
                    common_words = query_words.intersection(benchmark_words)
                    similarity = len(common_words) / max(len(query_words), len(benchmark_words))
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            'User Query': row['User Query'],
                            'SQL Query': row['SQL Query'],
                            'Classification': row['Classification'],
                            'similarity': similarity
                        }
            
            if best_similarity > 0.3:
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding best match: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_queries': len(self.benchmark_data),
            'classifications': self.benchmark_data['Classification'].value_counts().to_dict(),
            'avg_query_length': self.benchmark_data['User Query'].str.len().mean()
        }

agent = None

def create_app() -> FastAPI:
    app = FastAPI(title="beat it just beat it", version="1.0")
    
    @app.post("/evaluate")
    async def evaluate_sql(request: dict):
        try:
            if agent is None:
                raise HTTPException(status_code=500, detail="Agent not initialized")
            
            result = agent.evaluate(
                user_query=request.get('user_query', ''),
                generated_sql=request.get('generated_sql', '')
            )
            
            return {
                "rating": result.rating,
                "score": result.score,
                "explanation": result.explanation,
                "comparison_details": result.comparison_details,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/stats")
    async def get_stats():
        if agent is None:
            raise HTTPException(status_code=500, detail="Agent not initialized")
        
        return agent.get_stats()
    
    return app

workspace_url = os.getenv('DATABRICKS_HOST')
token = os.getenv('DATABRICKS_TOKEN')   
benchmark_file_path = os.getenv('BENCHMARK_FILE_PATH') 
    
agent = SQLEvaluationAgent(
workspace_url=workspace_url,
token=token,
benchmark_file_path=benchmark_file_path
)
logger.info("Agent initialized successfully")
    
app = create_app()
uvicorn.run(app, host="0.0.0.0", port=8000)