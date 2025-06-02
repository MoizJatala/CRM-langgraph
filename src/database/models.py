"""
SQLAlchemy database models for the CRM Agent system.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class TaskHistory(Base):
    """Model for storing task execution history."""
    __tablename__ = "task_history"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255), unique=True, index=True, nullable=False)
    user_query = Column(Text, nullable=False)
    task_type = Column(String(50), nullable=False)
    payload = Column(JSON, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    execution_time = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<TaskHistory(id={self.id}, task_id='{self.task_id}', status='{self.status}')>"


class ContactCache(Base):
    """Model for caching contact information."""
    __tablename__ = "contact_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    hubspot_id = Column(String(255), unique=True, index=True, nullable=False)
    email = Column(String(255), index=True, nullable=False)
    firstname = Column(String(255), nullable=True)
    lastname = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    company = Column(String(255), nullable=True)
    website = Column(String(255), nullable=True)
    jobtitle = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_synced = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<ContactCache(id={self.id}, email='{self.email}', hubspot_id='{self.hubspot_id}')>"


class DealCache(Base):
    """Model for caching deal information."""
    __tablename__ = "deal_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    hubspot_id = Column(String(255), unique=True, index=True, nullable=False)
    dealname = Column(String(255), nullable=False)
    amount = Column(Float, nullable=True)
    dealstage = Column(String(255), nullable=True)
    closedate = Column(DateTime(timezone=True), nullable=True)
    pipeline = Column(String(255), nullable=True)
    hubspot_owner_id = Column(String(255), nullable=True)
    associated_contact_email = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_synced = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<DealCache(id={self.id}, dealname='{self.dealname}', hubspot_id='{self.hubspot_id}')>"


class AgentExecution(Base):
    """Model for tracking individual agent executions."""
    __tablename__ = "agent_execution"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255), index=True, nullable=False)
    agent_name = Column(String(100), nullable=False)
    success = Column(Boolean, nullable=False)
    execution_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    execution_time = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<AgentExecution(id={self.id}, agent='{self.agent_name}', success={self.success})>"