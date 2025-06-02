"""
Pydantic models for the CRM Agent system.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, EmailStr
from enum import Enum


class TaskType(str, Enum):
    """Enumeration of supported task types."""
    CREATE_CONTACT = "create_contact"
    UPDATE_CONTACT = "update_contact"
    CREATE_DEAL = "create_deal"
    UPDATE_DEAL = "update_deal"
    SEND_EMAIL = "send_email"


class TaskStatus(str, Enum):
    """Enumeration of task statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ContactData(BaseModel):
    """Model for contact information."""
    email: EmailStr
    firstname: Optional[str] = None
    lastname: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    website: Optional[str] = None
    jobtitle: Optional[str] = None
    
    class Config:
        json_encoders = {
            EmailStr: str
        }


class DealData(BaseModel):
    """Model for deal information."""
    dealname: str
    amount: Optional[float] = None
    dealstage: Optional[str] = None
    closedate: Optional[datetime] = None
    pipeline: Optional[str] = None
    hubspot_owner_id: Optional[str] = None
    associated_contact_email: Optional[EmailStr] = None


class EmailData(BaseModel):
    """Model for email information."""
    to_email: EmailStr
    subject: str
    body: str
    html_body: Optional[str] = None


class UserQuery(BaseModel):
    """Model for user input queries."""
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class TaskRequest(BaseModel):
    """Model for task requests."""
    task_type: TaskType
    payload: Dict[str, Any]
    user_query: str
    priority: Optional[int] = Field(default=1, ge=1, le=5)


class TaskResponse(BaseModel):
    """Model for task responses."""
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class HubSpotContact(BaseModel):
    """Model for HubSpot contact response."""
    id: str
    properties: ContactData
    created_at: datetime
    updated_at: datetime


class HubSpotDeal(BaseModel):
    """Model for HubSpot deal response."""
    id: str
    properties: DealData
    created_at: datetime
    updated_at: datetime


class AgentResponse(BaseModel):
    """Model for agent responses."""
    agent_name: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


class OrchestrationResult(BaseModel):
    """Model for orchestration results."""
    task_id: str
    user_query: str
    identified_tasks: List[TaskRequest]
    results: List[AgentResponse]
    overall_success: bool
    total_execution_time: float
    summary: str