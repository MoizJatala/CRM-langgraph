"""
Data models for the CRM Agent System.
"""

from .schemas import *

__all__ = [
    "TaskType",
    "TaskStatus", 
    "ContactData",
    "DealData",
    "EmailData",
    "UserQuery",
    "TaskRequest",
    "TaskResponse",
    "AgentResponse",
    "OrchestrationResult"
]