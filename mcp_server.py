"""
FastMCP server implementation for the chatbot app.

This MCP server exposes the same functionality as the Streamlit app
but through the Model Context Protocol (MCP) using FastMCP framework.
"""
import os
import logging
from typing import List, Dict, Optional, Any
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from model_serving_utils import (
    endpoint_supports_feedback,
    query_endpoint,
    query_endpoint_stream,
    _get_endpoint_task_type,
    submit_feedback,
)
from messages import filter_clean_messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("mcp")

# Get serving endpoint from environment
SERVING_ENDPOINT = os.getenv('SERVING_ENDPOINT')
if not SERVING_ENDPOINT:
    raise ValueError(
        "Unable to determine serving endpoint. Set the SERVING_ENDPOINT environment variable."
    )

ENDPOINT_SUPPORTS_FEEDBACK = endpoint_supports_feedback(SERVING_ENDPOINT)

# In-memory chat history storage (in production, use a proper database)
chat_history: List[Dict[str, Any]] = []


# Pydantic models for request/response schemas
class ChatMessage(BaseModel):
    """Single chat message model."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls if applicable")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID for tool responses")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message to send to the chatbot")
    return_traces: bool = Field(False, description="Whether to return traces for feedback")
    stream: bool = Field(False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    messages: List[Dict[str, Any]] = Field(..., description="Assistant response messages")
    request_id: Optional[str] = Field(None, description="Request ID for feedback submission")
    session_id: Optional[str] = Field(None, description="Session ID for tracking conversation")


class FeedbackRequest(BaseModel):
    """Request model for feedback submission."""
    request_id: str = Field(..., description="Request ID from chat response")
    rating: int = Field(..., description="Feedback rating: 1 for positive, -1 for negative")


def reduce_chat_agent_chunks(chunks):
    """
    Reduce a list of ChatAgentChunk objects corresponding to a particular
    message into a single ChatAgentMessage
    """
    from mlflow.types.agent import ChatAgentChunk
    from collections import OrderedDict
    
    deltas = [chunk.delta for chunk in chunks]
    if not deltas:
        return None
    
    first_delta = deltas[0]
    result_msg = first_delta
    msg_contents = []
    
    # Accumulate tool calls properly
    tool_call_map = {}
    
    for delta in deltas:
        # Handle content
        if delta.content:
            msg_contents.append(delta.content)
            
        # Handle tool calls
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            for tool_call in delta.tool_calls:
                call_id = getattr(tool_call, 'id', None)
                tool_type = getattr(tool_call, 'type', "function")
                function_info = getattr(tool_call, 'function', None)
                if function_info:
                    func_name = getattr(function_info, 'name', "")
                    func_args = getattr(function_info, 'arguments', "")
                else:
                    func_name = ""
                    func_args = ""
                
                if call_id:
                    if call_id not in tool_call_map:
                        tool_call_map[call_id] = {
                            "id": call_id,
                            "type": tool_type,
                            "function": {
                                "name": func_name,
                                "arguments": func_args
                            }
                        }
                    else:
                        existing_args = tool_call_map[call_id]["function"]["arguments"]
                        tool_call_map[call_id]["function"]["arguments"] = existing_args + func_args
                        if func_name:
                            tool_call_map[call_id]["function"]["name"] = func_name

        # Handle tool call IDs
        if hasattr(delta, 'tool_call_id') and delta.tool_call_id:
            result_msg = result_msg.model_copy(update={"tool_call_id": delta.tool_call_id})
    
    # Convert tool call map back to list
    if tool_call_map:
        accumulated_tool_calls = list(tool_call_map.values())
        result_msg = result_msg.model_copy(update={"tool_calls": accumulated_tool_calls})
    
    result_msg = result_msg.model_copy(update={"content": "".join(msg_contents)})
    return result_msg


def query_chat_completions_endpoint(input_messages: List[Dict], return_traces: bool, stream: bool):
    """Handle ChatCompletions format."""
    if stream:
        accumulated_content = ""
        request_id = None
        
        for chunk in query_endpoint_stream(
            endpoint_name=SERVING_ENDPOINT,
            messages=input_messages,
            return_traces=return_traces
        ):
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    accumulated_content += content
            
            if "databricks_output" in chunk:
                req_id = chunk["databricks_output"].get("databricks_request_id")
                if req_id:
                    request_id = req_id
        
        if accumulated_content:
            return [{"role": "assistant", "content": accumulated_content}], request_id
        return [], request_id
    else:
        messages, request_id = query_endpoint(
            endpoint_name=SERVING_ENDPOINT,
            messages=input_messages,
            return_traces=return_traces
        )
        return messages, request_id


def query_chat_agent_endpoint(input_messages: List[Dict], return_traces: bool, stream: bool):
    """Handle ChatAgent streaming format."""
    from mlflow.types.agent import ChatAgentChunk
    from collections import OrderedDict
    
    if stream:
        message_buffers = OrderedDict()
        request_id = None
        
        for raw_chunk in query_endpoint_stream(
            endpoint_name=SERVING_ENDPOINT,
            messages=input_messages,
            return_traces=return_traces
        ):
            chunk = ChatAgentChunk.model_validate(raw_chunk)
            delta = chunk.delta
            message_id = delta.id

            req_id = raw_chunk.get("databricks_output", {}).get("databricks_request_id")
            if req_id:
                request_id = req_id
            
            if message_id not in message_buffers:
                message_buffers[message_id] = {"chunks": []}
            message_buffers[message_id]["chunks"].append(chunk)
        
        messages = []
        for msg_id, msg_info in message_buffers.items():
            reduced = reduce_chat_agent_chunks(msg_info["chunks"])
            if reduced:
                messages.append(reduced)
        
        final_messages = [msg.model_dump_compat(exclude_none=True) for msg in messages]
        clean_messages = filter_clean_messages(final_messages)
        return clean_messages, request_id
    else:
        messages, request_id = query_endpoint(
            endpoint_name=SERVING_ENDPOINT,
            messages=input_messages,
            return_traces=return_traces
        )
        return messages, request_id


def query_responses_endpoint(input_messages: List[Dict], return_traces: bool, stream: bool):
    """Handle ResponsesAgent streaming format."""
    from mlflow.types.responses import ResponsesAgentStreamEvent
    
    if stream:
        all_messages = []
        request_id = None

        for raw_event in query_endpoint_stream(
            endpoint_name=SERVING_ENDPOINT,
            messages=input_messages,
            return_traces=return_traces
        ):
            if "databricks_output" in raw_event:
                req_id = raw_event["databricks_output"].get("databricks_request_id")
                if req_id:
                    request_id = req_id
            
            if "type" in raw_event:
                event = ResponsesAgentStreamEvent.model_validate(raw_event)
                
                if hasattr(event, 'item') and event.item:
                    item = event.item
                    
                    if item.get("type") == "message":
                        content_parts = item.get("content", [])
                        for content_part in content_parts:
                            if content_part.get("type") == "output_text":
                                text = content_part.get("text", "")
                                if text:
                                    all_messages.append({
                                        "role": "assistant",
                                        "content": text
                                    })
                    
                    elif item.get("type") == "function_call":
                        call_id = item.get("call_id")
                        function_name = item.get("name")
                        arguments = item.get("arguments", "")
                        
                        all_messages.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": arguments
                                }
                            }]
                        })
                    
                    elif item.get("type") == "function_call_output":
                        call_id = item.get("call_id")
                        output = item.get("output", "")
                        
                        all_messages.append({
                            "role": "tool",
                            "content": output,
                            "tool_call_id": call_id
                        })

        clean_messages = filter_clean_messages(all_messages)
        return clean_messages, request_id
    else:
        messages, request_id = query_endpoint(
            endpoint_name=SERVING_ENDPOINT,
            messages=input_messages,
            return_traces=return_traces
        )
        return messages, request_id


def query_endpoint_and_handle(task_type: str, input_messages: List[Dict], return_traces: bool, stream: bool):
    """Handle streaming response based on task type."""
    if task_type == "agent/v1/responses":
        return query_responses_endpoint(input_messages, return_traces, stream)
    elif task_type == "agent/v2/chat":
        return query_chat_agent_endpoint(input_messages, return_traces, stream)
    else:  # chat/completions
        return query_chat_completions_endpoint(input_messages, return_traces, stream)


@mcp.tool()
def chat(message: str, return_traces: bool = False, stream: bool = False) -> Dict[str, Any]:
    """
    Send a message to the chatbot and get a response.
    
    This tool replicates the main chat functionality of the Streamlit app.
    It sends a user message to the Databricks serving endpoint and returns
    the assistant's response.
    
    Args:
        message: The user's message to send to the chatbot
        return_traces: Whether to return traces for feedback (default: False)
        stream: Whether to stream the response (default: False)
    
    Returns:
        Dictionary containing:
        - messages: List of assistant response messages
        - request_id: Request ID for feedback submission (if available)
        - session_id: Session ID for tracking conversation
    """
    global chat_history
    
    # Add user message to history
    user_msg = {"role": "user", "content": message}
    chat_history.append(user_msg)
    
    # Get the task type for this endpoint
    task_type = _get_endpoint_task_type(SERVING_ENDPOINT)
    
    # Convert history to standard chat message format
    input_messages = [msg for msg in chat_history]
    
    try:
        # Query endpoint and get response
        messages, request_id = query_endpoint_and_handle(
            task_type, input_messages, return_traces, stream
        )
        
        # Add assistant response to history
        for msg in messages:
            chat_history.append(msg)
        
        return {
            "messages": messages,
            "request_id": request_id,
            "session_id": "default"  # In production, use proper session management
        }
    except Exception as e:
        logger.error(f"Error querying endpoint: {e}")
        raise


@mcp.tool()
def get_chat_history() -> List[Dict[str, Any]]:
    """
    Get the current chat history.
    
    Returns:
        List of all messages in the chat history
    """
    return chat_history


@mcp.tool()
def clear_chat_history() -> Dict[str, str]:
    """
    Clear the chat history.
    
    Returns:
        Confirmation message
    """
    global chat_history
    chat_history = []
    return {"status": "success", "message": "Chat history cleared"}


@mcp.tool()
def submit_chat_feedback(request_id: str, rating: int) -> Dict[str, Any]:
    """
    Submit feedback for a chat response.
    
    This tool allows you to provide feedback (positive or negative) on
    an assistant's response, similar to the Streamlit app's feedback feature.
    
    Args:
        request_id: The request ID from the chat response
        rating: Feedback rating (1 for positive, -1 for negative)
    
    Returns:
        Dictionary with feedback submission status
    """
    if not ENDPOINT_SUPPORTS_FEEDBACK:
        return {"status": "error", "message": "Endpoint does not support feedback"}
    
    try:
        submit_feedback(
            endpoint=SERVING_ENDPOINT,
            request_id=request_id,
            rating=rating
        )
        return {
            "status": "success",
            "message": f"Feedback submitted: {'positive' if rating == 1 else 'negative'}"
        }
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
def get_endpoint_info() -> Dict[str, Any]:
    """
    Get information about the serving endpoint.
    
    Returns:
        Dictionary with endpoint configuration information
    """
    task_type = _get_endpoint_task_type(SERVING_ENDPOINT)
    return {
        "endpoint_name": SERVING_ENDPOINT,
        "task_type": task_type,
        "supports_feedback": ENDPOINT_SUPPORTS_FEEDBACK
    }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()

