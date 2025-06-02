"""
Database modules for the CRM Agent System.
"""

from .connection import get_db, create_tables, db_manager
from .models import Base, TaskHistory, ContactCache, DealCache, AgentExecution

__all__ = [
    "get_db",
    "create_tables", 
    "db_manager",
    "Base",
    "TaskHistory",
    "ContactCache",
    "DealCache",
    "AgentExecution"
]