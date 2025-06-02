"""
LangGraph API-compatible graphs for the HubSpot CRM system.
This module exports the graphs at the root level to avoid package name conflicts.
"""

from src.agents.email_agent import langgraph_api_email_graph
from src.agents.hubspot_agent import langgraph_api_hubspot_graph  
from src.agents.orchestrator_agent import langgraph_api_orchestrator_graph

# Export graphs for LangGraph API
email_agent_graph = langgraph_api_email_graph
hubspot_agent_graph = langgraph_api_hubspot_graph
orchestrator_agent_graph = langgraph_api_orchestrator_graph

__all__ = [
    "email_agent_graph",
    "hubspot_agent_graph", 
    "orchestrator_agent_graph"
] 