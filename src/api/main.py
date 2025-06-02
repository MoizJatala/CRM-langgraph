"""
FastAPI application for the CRM Agent system.
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import asyncio
from typing import Dict, Any, List

from src.config.settings import settings
from src.database.connection import get_db, create_tables
from src.database.models import TaskHistory, AgentExecution
from src.models.schemas import (
    UserQuery, TaskResponse, OrchestrationResult, 
    ContactData, DealData, EmailData
)
from src.agents.orchestrator_agent import GlobalOrchestratorAgent
from src.agents.hubspot_agent import HubSpotAgent
from src.agents.email_agent import EmailAgent
from src.utils.logger import configure_logging, get_logger
from src.utils.helpers import generate_task_id, current_timestamp

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CRM Agent System",
    description="Multi-agent AI system for CRM operations using LangGraph",
    version="1.0.0",
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
orchestrator = GlobalOrchestratorAgent()
hubspot_agent = HubSpotAgent()
email_agent = EmailAgent()


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting CRM Agent System")
    
    # Create database tables
    create_tables()
    logger.info("Database tables created")
    
    logger.info("CRM Agent System started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down CRM Agent System")


@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "CRM Agent System",
        "version": "1.0.0",
        "status": "running",
        "agents": ["GlobalOrchestrator", "HubSpotAgent", "EmailAgent"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": current_timestamp().isoformat(),
        "database": "connected"
    }


@app.post("/process-query", response_model=OrchestrationResult)
async def process_user_query(
    query: UserQuery,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Process a user query through the multi-agent system.
    """
    try:
        logger.info(f"Processing user query: {query.query}")
        
        # Process the query through the orchestrator
        result = await orchestrator.process_user_query(query.query)
        
        # Store task history in background
        background_tasks.add_task(
            store_task_history,
            result,
            db
        )
        
        logger.info(f"Query processed successfully: {result.task_id}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to process query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hubspot/create-contact")
async def create_hubspot_contact(
    contact_data: ContactData,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create a new contact in HubSpot.
    """
    try:
        logger.info(f"Creating HubSpot contact: {contact_data.email}")
        
        result = await hubspot_agent.process_task(
            "create_contact",
            contact_data.dict()
        )
        
        # Store execution history
        background_tasks.add_task(
            store_agent_execution,
            generate_task_id(),
            result,
            db
        )
        
        if result.success:
            return {"status": "success", "data": result.data}
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        logger.error(f"Failed to create contact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hubspot/update-contact")
async def update_hubspot_contact(
    contact_data: ContactData,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Update an existing contact in HubSpot.
    """
    try:
        logger.info(f"Updating HubSpot contact: {contact_data.email}")
        
        result = await hubspot_agent.process_task(
            "update_contact",
            contact_data.dict()
        )
        
        # Store execution history
        background_tasks.add_task(
            store_agent_execution,
            generate_task_id(),
            result,
            db
        )
        
        if result.success:
            return {"status": "success", "data": result.data}
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        logger.error(f"Failed to update contact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hubspot/create-deal")
async def create_hubspot_deal(
    deal_data: DealData,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create a new deal in HubSpot.
    """
    try:
        logger.info(f"Creating HubSpot deal: {deal_data.dealname}")
        
        result = await hubspot_agent.process_task(
            "create_deal",
            deal_data.dict()
        )
        
        # Store execution history
        background_tasks.add_task(
            store_agent_execution,
            generate_task_id(),
            result,
            db
        )
        
        if result.success:
            return {"status": "success", "data": result.data}
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        logger.error(f"Failed to create deal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/email/send-notification")
async def send_email_notification(
    email_data: EmailData,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Send an email notification.
    """
    try:
        logger.info(f"Sending email to: {email_data.to_email}")
        
        result = await email_agent.process_task(
            "send_email",
            email_data.dict()
        )
        
        # Store execution history
        background_tasks.add_task(
            store_agent_execution,
            generate_task_id(),
            result,
            db
        )
        
        if result.success:
            return {"status": "success", "data": result.data}
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/history")
async def get_task_history(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Get task execution history.
    """
    try:
        tasks = db.query(TaskHistory)\
                 .order_by(TaskHistory.created_at.desc())\
                 .offset(offset)\
                 .limit(limit)\
                 .all()
        
        return {
            "tasks": [
                {
                    "task_id": task.task_id,
                    "user_query": task.user_query,
                    "task_type": task.task_type,
                    "status": task.status,
                    "created_at": task.created_at,
                    "execution_time": task.execution_time
                }
                for task in tasks
            ],
            "total": len(tasks)
        }
        
    except Exception as e:
        logger.error(f"Failed to get task history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/{task_id}")
async def get_task_details(
    task_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific task.
    """
    try:
        task = db.query(TaskHistory).filter(TaskHistory.task_id == task_id).first()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Get agent executions for this task
        executions = db.query(AgentExecution)\
                      .filter(AgentExecution.task_id == task_id)\
                      .all()
        
        return {
            "task": {
                "task_id": task.task_id,
                "user_query": task.user_query,
                "task_type": task.task_type,
                "status": task.status,
                "result": task.result,
                "error_message": task.error_message,
                "execution_time": task.execution_time,
                "created_at": task.created_at,
                "completed_at": task.completed_at
            },
            "agent_executions": [
                {
                    "agent_name": execution.agent_name,
                    "success": execution.success,
                    "execution_data": execution.execution_data,
                    "error_message": execution.error_message,
                    "execution_time": execution.execution_time,
                    "created_at": execution.created_at
                }
                for execution in executions
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/status")
async def get_agents_status():
    """
    Get status of all agents.
    """
    return {
        "orchestrator": {
            "name": "GlobalOrchestratorAgent",
            "status": "active",
            "capabilities": ["task_delegation", "query_analysis", "result_summarization"]
        },
        "hubspot": {
            "name": "HubSpotAgent",
            "status": "active",
            "capabilities": ["create_contact", "update_contact", "create_deal", "update_deal"]
        },
        "email": {
            "name": "EmailAgent",
            "status": "active",
            "capabilities": ["send_notification", "generate_content"]
        }
    }


# Background task functions
def store_task_history(result: OrchestrationResult, db: Session):
    """Store task execution history in the database."""
    try:
        task_history = TaskHistory(
            task_id=result.task_id,
            user_query=result.user_query,
            task_type="orchestration",
            payload={"identified_tasks": [task.dict() for task in result.identified_tasks]},
            status="completed" if result.overall_success else "failed",
            result={"summary": result.summary, "execution_time": result.total_execution_time},
            execution_time=result.total_execution_time,
            completed_at=current_timestamp()
        )
        
        db.add(task_history)
        db.commit()
        
        # Store individual agent executions
        for agent_result in result.results:
            store_agent_execution(result.task_id, agent_result, db)
            
    except Exception as e:
        logger.error(f"Failed to store task history: {str(e)}")
        db.rollback()


def store_agent_execution(task_id: str, result: Any, db: Session):
    """Store individual agent execution in the database."""
    try:
        if hasattr(result, 'agent_name'):
            execution = AgentExecution(
                task_id=task_id,
                agent_name=result.agent_name,
                success=result.success,
                execution_data=result.data,
                error_message=result.error,
                execution_time=result.execution_time
            )
            
            db.add(execution)
            db.commit()
            
    except Exception as e:
        logger.error(f"Failed to store agent execution: {str(e)}")
        db.rollback()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )