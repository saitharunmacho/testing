import json
import operator
from typing import Annotated, Any, Sequence, TypedDict, Literal, Generator
from uuid import uuid4
import warnings

import mlflow
from databricks.sdk import WorkspaceClient
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from openai import OpenAI

############################################
# Configuration
############################################
LLM_ENDPOINT_NAME = "databricks-gpt-oss-120b"

GENIE_SPACES = [
    {
        "id": "genie_space_1",
        "name": "DataAnalysisAgent",
        "description": "Specializes in data analysis, SQL queries, and data visualization tasks",
        "space_id": "01j0abc123def"  # TODO: Replace with your Genie Space ID
    },
    {
        "id": "genie_space_2",
        "name": "MLOpsAgent",
        "description": "Handles machine learning operations, model training, and deployment",
        "space_id": "01j0abc456ghi"  # TODO: Replace with your Genie Space ID
    },
    {
        "id": "genie_space_3",
        "name": "DataEngineeringAgent",
        "description": "Manages ETL pipelines, data engineering workflows, and data quality",
        "space_id": "01j0abc789jkl"  # TODO: Replace with your Genie Space ID
    },
    {
        "id": "genie_space_4",
        "name": "BusinessIntelligenceAgent",
        "description": "Provides business metrics, KPIs, and executive reporting",
        "space_id": "01j0abcmnopqr"  # TODO: Replace with your Genie Space ID
    },
    {
        "id": "genie_space_5",
        "name": "SecurityComplianceAgent",
        "description": "Handles security audits, compliance checks, and governance policies",
        "space_id": "01j0abcstuvwx"  # TODO: Replace with your Genie Space ID
    },
    {
        "id": "genie_space_6",
        "name": "CustomerSupportAgent",
        "description": "Answers customer queries, troubleshooting, and support documentation",
        "space_id": "01j0abcyz1234"  # TODO: Replace with your Genie Space ID
    }
]

SUPERVISOR_SYSTEM_PROMPT = f"""
You are a supervisor orchestrating a team of specialized agents. Your role is to:
1. Analyze the user's request
2. Route the request to the most appropriate specialist agent
3. Coordinate multiple agents if needed for complex tasks
4. Synthesize responses from multiple agents when necessary

Available agents:
{chr(10).join([f"- {agent['name']}: {agent['description']}" for agent in GENIE_SPACES])}

Always route to the most relevant agent. If the task requires multiple agents, coordinate them appropriately.
If uncertain, default to the most general agent or ask the user for clarification.
"""

############################################
# State Definition
############################################
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    task_complete: bool
    iteration_count: int

############################################
# Genie Space Integration
############################################
class GenieSpaceAgent:
    """Wrapper for calling a Genie Space"""
    
    def __init__(self, space_config: dict, workspace_client: WorkspaceClient):
        self.space_id = space_config["space_id"]
        self.name = space_config["name"]
        self.description = space_config["description"]
        self.workspace_client = workspace_client
    
    @mlflow.trace(span_type=SpanType.AGENT)
    def invoke(self, message: str) -> str:
        """
        Call the Genie Space with a message
        TODO: Implement actual Genie Space API call
        For now, this is a placeholder that simulates a response
        """
        try:
            # Placeholder for Genie Space API call
            # In production, you would use the Genie API:
            # response = self.workspace_client.genie.spaces.query(
            #     space_id=self.space_id,
            #     content=message
            # )
            
            # Simulated response for demonstration
            response = f"[{self.name}] Processing your request: {message[:100]}...\n"
            response += f"This is a simulated response from {self.name}. "
            response += f"In production, this would call Genie Space ID: {self.space_id}"
            return response
        except Exception as e:
            return f"Error calling {self.name}: {str(e)}"

############################################
# Supervisor Agent
############################################
class SupervisorAgent:
    """Supervisor that routes tasks to specialist agents"""
    
    def __init__(self, llm_client: OpenAI, genie_agents: dict):
        self.llm_client = llm_client
        self.genie_agents = genie_agents
    
    @mlflow.trace(span_type=SpanType.AGENT)
    def route_task(self, state: AgentState) -> AgentState:
        """Determine which agent should handle the task"""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        iteration_count = state.get("iteration_count", 0)
        
        # Create routing prompt
        agent_options = "\n".join([
            f"- {name}: {self.genie_agents[name].description}" 
            for name in self.genie_agents.keys()
        ])
        
        routing_prompt = f"""
Given the user request and conversation history, select the most appropriate agent to handle it.

User request: {last_message}

Available agents:
{agent_options}

Respond with ONLY the agent name (e.g., DataAnalysisAgent) or "FINISH" if the task is complete.
"""
        
        # Call LLM for routing decision
        response = self.llm_client.chat.completions.create(
            model=LLM_ENDPOINT_NAME,
            messages=[
                {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
                {"role": "user", "content": routing_prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        
        next_agent = response.choices[0].message.content.strip()
        
        # Validate the routing decision
        if next_agent == "FINISH" or next_agent not in self.genie_agents or iteration_count >= 10:
            return {
                **state, 
                "next_agent": "FINISH", 
                "task_complete": True,
                "iteration_count": iteration_count + 1
            }
        
        return {
            **state, 
            "next_agent": next_agent, 
            "task_complete": False,
            "iteration_count": iteration_count + 1
        }

############################################
# Agent Executor Node
############################################
def create_agent_node(agent_name: str, genie_agent: GenieSpaceAgent):
    """Create a node that executes a specific Genie agent"""
    
    @mlflow.trace(span_type=SpanType.AGENT)
    def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        # Call the Genie Space
        response = genie_agent.invoke(last_message)
        
        # Add response to messages
        new_message = AIMessage(content=response, name=agent_name)
        
        return {
            "messages": [new_message],
            "next_agent": "supervisor",
            "task_complete": False,
            "iteration_count": state.get("iteration_count", 0)
        }
    
    return agent_node

############################################
# Multi-Agent System
############################################
class MultiAgentSupervisorSystem(ResponsesAgent):
    """LangGraph-based multi-agent system with supervisor"""
    
    def __init__(self, llm_endpoint: str, genie_spaces: list):
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.llm_client = self.workspace_client.serving_endpoints.get_open_ai_client()
        
        # Initialize Genie agents
        self.genie_agents = {
            space["name"]: GenieSpaceAgent(space, self.workspace_client)
            for space in genie_spaces
        }
        
        # Initialize supervisor
        self.supervisor = SupervisorAgent(self.llm_client, self.genie_agents)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add supervisor node
        workflow.add_node("supervisor", self.supervisor.route_task)
        
        # Add agent nodes
        for agent_name, genie_agent in self.genie_agents.items():
            workflow.add_node(
                agent_name,
                create_agent_node(agent_name, genie_agent)
            )
        
        # Define conditional routing
        def should_continue(state: AgentState) -> str:
            if state.get("task_complete", False):
                return END
            next_agent = state.get("next_agent", "supervisor")
            if next_agent == "FINISH":
                return END
            return next_agent
        
        # Add edges from supervisor to all agents
        agent_routing = {agent_name: agent_name for agent_name in self.genie_agents.keys()}
        agent_routing[END] = END
        
        workflow.add_conditional_edges(
            "supervisor",
            should_continue,
            agent_routing
        )
        
        # Add edges from agents back to supervisor
        for agent_name in self.genie_agents.keys():
            workflow.add_edge(agent_name, "supervisor")
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        # Compile the graph
        return workflow.compile()
    
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Process a request through the multi-agent system"""
        # Convert input to messages
        input_messages = []
        for item in request.input:
            if item.role == "user":
                input_messages.append(HumanMessage(content=item.content))
            elif item.role == "assistant":
                input_messages.append(AIMessage(content=item.content))
        
        # Initialize state
        initial_state = {
            "messages": input_messages,
            "next_agent": "supervisor",
            "task_complete": False,
            "iteration_count": 0
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Extract final messages
        final_messages = result["messages"]
        
        # Convert back to ResponsesAgent format
        outputs = []
        for msg in final_messages:
            if isinstance(msg, AIMessage):
                outputs.append(
                    self.create_text_output_item(msg.content, str(uuid4()))
                )
        
        return ResponsesAgentResponse(
            output=outputs,
            custom_outputs=request.custom_inputs
        )
    
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream predictions through the multi-agent system"""
        # Convert input to messages
        input_messages = []
        for item in request.input:
            if item.role == "user":
                input_messages.append(HumanMessage(content=item.content))
            elif item.role == "assistant":
                input_messages.append(AIMessage(content=item.content))
        
        # Initialize state
        initial_state = {
            "messages": input_messages,
            "next_agent": "supervisor",
            "task_complete": False,
            "iteration_count": 0
        }
        
        # Stream through the graph
        for chunk in self.graph.stream(initial_state):
            for node_name, node_output in chunk.items():
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if isinstance(msg, AIMessage):
                            output_item = self.create_text_output_item(
                                msg.content, 
                                str(uuid4())
                            )
                            yield ResponsesAgentStreamEvent(
                                type="response.output_item.done",
                                item=output_item
                            )

############################################
# Initialize the Multi-Agent System
############################################
mlflow.openai.autolog()
AGENT = MultiAgentSupervisorSystem(
    llm_endpoint=LLM_ENDPOINT_NAME, 
    genie_spaces=GENIE_SPACES
)
mlflow.models.set_model(AGENT)