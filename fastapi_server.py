"""
FastAPI wrapper for the FastMCP chatbot server.

This FastAPI application wraps the FastMCP server functionality,
exposing the MCP tools as HTTP endpoints.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging

# Import mcp_server module - this initializes all the necessary variables
import mcp_server

# Import the functions directly (they're decorated with @mcp.tool but still callable as regular functions)
mcp_chat = mcp_server.chat
mcp_get_chat_history = mcp_server.get_chat_history
mcp_clear_chat_history = mcp_server.clear_chat_history
mcp_submit_chat_feedback = mcp_server.submit_chat_feedback
mcp_get_endpoint_info = mcp_server.get_endpoint_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="E2E Chatbot API",
    description="FastAPI wrapper for the FastMCP chatbot server",
    version="1.0.0"
)


# Request/Response models for FastAPI
class ChatRequestModel(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message to send to the chatbot")
    return_traces: bool = Field(False, description="Whether to return traces for feedback")
    stream: bool = Field(False, description="Whether to stream the response")


class ChatResponseModel(BaseModel):
    """Response model for chat endpoint."""
    messages: List[Dict[str, Any]] = Field(..., description="Assistant response messages")
    request_id: Optional[str] = Field(None, description="Request ID for feedback submission")
    session_id: Optional[str] = Field(None, description="Session ID for tracking conversation")


class FeedbackRequestModel(BaseModel):
    """Request model for feedback submission."""
    request_id: str = Field(..., description="Request ID from chat response")
    rating: int = Field(..., description="Feedback rating: 1 for positive, -1 for negative")


class FeedbackResponseModel(BaseModel):
    """Response model for feedback submission."""
    status: str = Field(..., description="Status of the feedback submission")
    message: str = Field(..., description="Status message")


class ClearHistoryResponseModel(BaseModel):
    """Response model for clear history endpoint."""
    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Status message")


class EndpointInfoResponseModel(BaseModel):
    """Response model for endpoint info endpoint."""
    endpoint_name: str = Field(..., description="Name of the serving endpoint")
    task_type: str = Field(..., description="Task type of the endpoint")
    supports_feedback: bool = Field(..., description="Whether the endpoint supports feedback")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "E2E Chatbot API",
        "description": "FastAPI wrapper for the FastMCP chatbot server",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/chat",
            "chat_history": "/api/chat/history",
            "clear_history": "/api/chat/history",
            "feedback": "/api/feedback",
            "endpoint_info": "/api/endpoint/info"
        }
    }


@app.post("/api/chat", response_model=ChatResponseModel)
async def chat_endpoint(request: ChatRequestModel):
    """
    Send a message to the chatbot and get a response.
    
    This endpoint replicates the main chat functionality of the Streamlit app.
    It sends a user message to the Databricks serving endpoint and returns
    the assistant's response.
    """
    try:
        result = mcp_chat(
            message=request.message,
            return_traces=request.return_traces,
            stream=request.stream
        )
        return ChatResponseModel(**result)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/history", response_model=List[Dict[str, Any]])
async def get_chat_history_endpoint():
    """
    Get the current chat history.
    
    Returns all messages in the current chat session.
    """
    try:
        return mcp_get_chat_history()
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/history", response_model=ClearHistoryResponseModel)
async def clear_chat_history_endpoint():
    """
    Clear the chat history.
    
    Removes all messages from the chat history.
    """
    try:
        result = mcp_clear_chat_history()
        return ClearHistoryResponseModel(**result)
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback", response_model=FeedbackResponseModel)
async def submit_feedback_endpoint(request: FeedbackRequestModel):
    """
    Submit feedback for a chat response.
    
    This endpoint allows you to provide feedback (positive or negative) on
    an assistant's response, similar to the Streamlit app's feedback feature.
    """
    try:
        result = mcp_submit_chat_feedback(
            request_id=request.request_id,
            rating=request.rating
        )
        return FeedbackResponseModel(**result)
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/endpoint/info", response_model=EndpointInfoResponseModel)
async def get_endpoint_info_endpoint():
    """
    Get information about the serving endpoint.
    
    Returns configuration information about the serving endpoint.
    """
    try:
        result = mcp_get_endpoint_info()
        return EndpointInfoResponseModel(**result)
    except Exception as e:
        logger.error(f"Error getting endpoint info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

