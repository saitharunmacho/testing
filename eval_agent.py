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
from dotenv import load_dotenv

load_dotenv()

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
        self.workspace_url = workspace_url
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
        self.workspace_url = workspace_url
        self.token = token
        self.model_name = model_name
        
        self.databricks = DatabricksClient(workspace_url, token)
        
        self.sql_comparator = SQLComparator(self.databricks)
        
        self.benchmark_data = self._load_benchmark_data(benchmark_file_path)
        
        logger.info(f"Agent initialized with {len(self.benchmark_data)} benchmark queries")
    
    def _load_benchmark_data(self, file_path: str) -> pd.DataFrame:
        try:
            logger.info(f"Loading benchmark data from: {file_path}")
            
            if file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.xls'):
                logger.info("Reading Excel file")
                data = pd.read_excel(file_path)
            else:
                logger.info("Reading CSV file")
                data = pd.read_csv(file_path)
            
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            logger.info(f"Columns: {list(data.columns)}")
            logger.info(f"Data types: {data.dtypes.to_dict()}")
            
            # Check for any list-like data in the first few rows
            logger.info("Checking first 5 rows for data types:")
            for idx in range(min(5, len(data))):
                row = data.iloc[idx]
                logger.info(f"Row {idx}:")
                for col in data.columns:
                    value = row[col]
                    logger.info(f"  {col}: {type(value)} = {repr(value)}")
                    if isinstance(value, list):
                        logger.warning(f"  WARNING: {col} at row {idx} is a list: {value}")
            
            required_columns = ['User Query', 'SQL Query']
            missing = [col for col in required_columns if col not in data.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            
            logger.info("Dropping rows with missing values")
            data = data.dropna(subset=required_columns)
            logger.info(f"Data shape after dropping NaN: {data.shape}")
            
            # Safely convert all columns to strings to handle any list or other non-string types
            logger.info("Converting columns to strings")
            for col in required_columns:
                logger.info(f"Converting column: {col}")
                original_dtype = data[col].dtype
                logger.info(f"  Original dtype: {original_dtype}")
                
                # Check for any list values before conversion
                list_count = 0
                for idx, value in data[col].items():
                    if isinstance(value, list):
                        list_count += 1
                        logger.warning(f"  List found at row {idx}: {value}")
                
                if list_count > 0:
                    logger.warning(f"  Found {list_count} list values in column {col}")
                
                data[col] = data[col].astype(str).str.strip()
                logger.info(f"  Converted to string dtype: {data[col].dtype}")
            
            logger.info("Data loading completed successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error loading benchmark data: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def evaluate(
        self,
        user_query: str,
        generated_sql: str
    ) -> EvaluationResult:

        try:
            logger.info(f"Starting evaluation for user query: {user_query}")
            logger.info(f"Generated SQL: {generated_sql}")
            
            logger.info("Finding best match...")
            best_match = self._find_best_match(user_query)
            
            if best_match is None:
                logger.info("No matching benchmark query found")
                return EvaluationResult(
                    rating="Bad",
                    explanation="No matching benchmark query found",
                    score=0.0,
                    comparison_details={'error': 'No benchmark found'}
                )
            
            logger.info(f"Best match found: {best_match}")
            benchmark_sql = best_match['SQL Query']
            match_info = best_match
            
            logger.info("Evaluating SQL comparison...")
            comparison = self.sql_comparator.evaluate_sql_comparison(generated_sql, benchmark_sql)
            logger.info(f"Comparison result: {comparison}")
            
            result = EvaluationResult(
                rating=comparison['rating'],
                explanation=comparison['explanation'],
                score=comparison['score'],
                comparison_details={
                    'rule_applied': comparison['rule_applied'],
                    'details': comparison['details'],
                    'match_info': match_info
                }
            )
            
            logger.info(f"Evaluation completed successfully. Rating: {result.rating}, Score: {result.score}")
            return result
            
        except Exception as e:  
            logger.error(f"Evaluation error: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
            
            logger.info(f"Starting best match search for query: {user_query}")
            logger.info(f"Benchmark data shape: {self.benchmark_data.shape}")
            logger.info(f"Benchmark data columns: {list(self.benchmark_data.columns)}")
            
            for idx, row in self.benchmark_data.iterrows():
                try:
                    logger.debug(f"Processing row {idx}")
                    
                    # Debug the raw data types
                    user_query_raw = row['User Query']
                    sql_query_raw = row['SQL Query']
                    
                    logger.debug(f"Raw data types - User Query: {type(user_query_raw)}, SQL Query: {type(sql_query_raw)}")
                    logger.debug(f"Raw values - User Query: {repr(user_query_raw)}, SQL Query: {repr(sql_query_raw)}")
                    
                    # Check if any value is a list
                    if isinstance(user_query_raw, list):
                        logger.warning(f"User Query at row {idx} is a list: {user_query_raw}")
                    if isinstance(sql_query_raw, list):
                        logger.warning(f"SQL Query at row {idx} is a list: {sql_query_raw}")
                    
                    # Safely convert values to strings to handle any list or other non-string types
                    user_query_benchmark = str(user_query_raw) if pd.notna(user_query_raw) else ""
                    sql_query_benchmark = str(sql_query_raw) if pd.notna(sql_query_raw) else ""
                    
                    logger.debug(f"Converted values - User Query: {repr(user_query_benchmark)}, SQL Query: {repr(sql_query_benchmark)}")
                    
                    # Try to create sets for comparison
                    try:
                        query_words = set(user_query.lower().split())
                        logger.debug(f"Query words created successfully: {query_words}")
                    except Exception as e:
                        logger.error(f"Error creating query_words set: {e}")
                        logger.error(f"User query that caused error: {repr(user_query)}")
                        raise
                    
                    try:
                        benchmark_words = set(user_query_benchmark.lower().split())
                        logger.debug(f"Benchmark words created successfully: {benchmark_words}")
                    except Exception as e:
                        logger.error(f"Error creating benchmark_words set: {e}")
                        logger.error(f"User query benchmark that caused error: {repr(user_query_benchmark)}")
                        raise
                    
                    if query_words and benchmark_words:
                        common_words = query_words.intersection(benchmark_words)
                        similarity = len(common_words) / max(len(query_words), len(benchmark_words))
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = {
                                'User Query': user_query_benchmark,
                                'SQL Query': sql_query_benchmark,
                                'similarity': similarity
                            }
                            logger.debug(f"New best match found with similarity: {similarity}")
                
                except Exception as row_error:
                    logger.error(f"Error processing row {idx}: {row_error}")
                    logger.error(f"Row data: {dict(row)}")
                    continue
            
            logger.info(f"Best match search completed. Best similarity: {best_similarity}")
            if best_similarity > 0.3:
                logger.info(f"Returning best match with similarity: {best_similarity}")
                return best_match
            else:
                logger.info("No match found above threshold (0.3)")
                return None
            
        except Exception as e:
            logger.error(f"Error finding best match: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            return {
                'total_queries': len(self.benchmark_data),
                'avg_query_length': self.benchmark_data['User Query'].str.len().mean()
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'total_queries': 0,
                'avg_query_length': 0.0,
                'error': str(e)
            }

agent = None


def create_app() -> FastAPI:
    app = FastAPI(title="beat it just beat it", version="1.0")
    
    @app.post("/evaluate")
    async def evaluate_sql(request: dict):
        try:
            logger.info(f"Received evaluation request: {request}")
            
            if agent is None:
                logger.error("Agent not initialized")
                raise HTTPException(status_code=500, detail="Agent not initialized")
            
            user_query1 = request.get('user_query', '')
            logger.info(f"User query: {user_query1}")
            
            logger.info("Calling Genie space...")
            input_to_genie = querying_space(user_query1)
            logger.info(f"Genie response: {input_to_genie}")
            
            logger.info("Getting SQL response...")
            message_data = input_to_genie.get('message', {})
            conversation_id = input_to_genie.get('conversation_id')
            message_id = message_data.get('message_id') or input_to_genie.get('message_id')
            logger.info(f"Conversation ID: {conversation_id}")
            logger.info(f"Message ID: {message_id}")
            raw_generated_sql = getting_response(conversation_id, message_id)
            generated_sql1 = raw_generated_sql.get('query')
            logger.info(f"Generated SQL: {generated_sql1}")
            
            logger.info("Starting agent evaluation...")
            result = agent.evaluate(
                user_query1,
                generated_sql1                
            )
            logger.info(f"Evaluation result: {result}")
            
            return {
                "rating": result.rating,
                "score": result.score,
                "explanation": result.explanation,
                "comparison_details": result.comparison_details,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
genie_space_id = os.getenv('GENIE_SPACE_ID')
base_url = f"{workspace_url}/api/2.0/genie"
headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
}

def querying_space(input_message):
    url = f"{base_url}/spaces/{genie_space_id}/start-conversation"
    data = {
        "content": input_message
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()


def getting_response(conversation_id, message_id):
    try:
        import time
        
        # Poll for message completion with timeout
        max_wait_time = 60  # 60 seconds timeout
        poll_interval = 2   # Poll every 2 seconds
        elapsed_time = 0
        
        logger.info(f"Starting to poll for message completion. Message ID: {message_id}")
        
        while elapsed_time < max_wait_time:
            # Get the conversation messages
            url = f"{base_url}/spaces/{genie_space_id}/conversations/{conversation_id}/messages"
            logger.info(f"Polling messages from: {url} (elapsed: {elapsed_time}s)")
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            messages_data = response.json()
            
            messages = messages_data.get('messages', [])
            if not messages:
                logger.warning("No messages found in response")
                time.sleep(poll_interval)
                elapsed_time += poll_interval
                continue
            
            # Find the latest message
            latest_message = messages[-1]
            status = latest_message.get('status')
            logger.info(f"Message status: {status}")
            
            if status == "COMPLETED":
                logger.info("Message completed! Looking for SQL query...")
                
                # Look for SQL query in attachments
                attachments = latest_message.get('attachments', [])
                logger.info(f"Attachments found: {len(attachments)}")
                logger.info(f"Attachments: {attachments}")
                
                if attachments:
                    for attachment in attachments:
                        # Prioritize query field for SQL
                        if attachment.get('query'):
                            sql_query = attachment.get('query')
                            logger.info(f"Found SQL query: {sql_query}")
                            return sql_query
                        # Fallback to text if it contains SQL-like content
                        if attachment.get('text'):
                            text = attachment.get('text')
                            # Check if text contains SQL keywords
                            if any(keyword in text.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE']):
                                logger.info(f"Found SQL in text: {text}")
                                return text
                
                # Check message content as fallback
                content = latest_message.get('content', '')
                if content and any(keyword in content.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE']):
                    logger.info(f"Found SQL in message content: {content}")
                    return content
                
                logger.warning("Message completed but no SQL query found in attachments or content")
                return "No response found"
                
            elif status in ["FAILED", "CANCELLED"]:
                logger.error(f"Message failed with status: {status}")
                return "No response found"
            
            # Message is still processing, wait and try again
            logger.info(f"Message still processing (status: {status}), waiting {poll_interval}s...")
            time.sleep(poll_interval)
            elapsed_time += poll_interval
        
        logger.warning(f"Timeout reached ({max_wait_time}s) while waiting for message completion")
        return "No response found"
        
    except Exception as e:
        logger.error(f"Error getting response: {e}")
        return f"Error: {str(e)}"

agent = SQLEvaluationAgent(
workspace_url=workspace_url,
token=token,
benchmark_file_path=benchmark_file_path
)
logger.info("Agent initialized successfully")
    
app = create_app()
uvicorn.run(app, host="0.0.0.0", port=8000)
