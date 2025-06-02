"""
LangGraph API-compatible graphs for the HubSpot CRM system.
This module is isolated from the main src package to avoid naming conflicts.
"""

import sys
import os

# Add the parent directory to the Python path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the full agent graphs (not the simplified API-compatible ones)
from src.agents.email_agent import email_agent_graph as full_email_graph
from src.agents.hubspot_agent import hubspot_agent_graph as full_hubspot_graph
from src.agents.orchestrator_agent import orchestrator_agent_graph as full_orchestrator_graph

# Export the full graphs for LangGraph Studio visualization
email_agent_graph = full_email_graph
hubspot_agent_graph = full_hubspot_graph
orchestrator_agent_graph = full_orchestrator_graph

__all__ = [
    "email_agent_graph",
    "hubspot_agent_graph", 
    "orchestrator_agent_graph"
] 