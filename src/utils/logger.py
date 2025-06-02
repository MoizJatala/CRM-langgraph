"""
Logging utilities for the CRM Agent system.
"""
import logging
import structlog
from typing import Any, Dict
from src.config.settings import settings


def configure_logging():
    """Configure structured logging for the application."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper()),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


class AgentLogger:
    """Logger wrapper for agent operations."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = get_logger(agent_name)
    
    def info(self, message: str, **kwargs):
        """Log info message with agent context."""
        self.logger.info(message, agent=self.agent_name, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with agent context."""
        self.logger.error(message, agent=self.agent_name, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with agent context."""
        self.logger.warning(message, agent=self.agent_name, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with agent context."""
        self.logger.debug(message, agent=self.agent_name, **kwargs)
    
    def log_task_start(self, task_id: str, task_type: str, **kwargs):
        """Log task start with structured data."""
        self.info(
            "Task started",
            task_id=task_id,
            task_type=task_type,
            **kwargs
        )
    
    def log_task_completion(self, task_id: str, success: bool, execution_time: float, **kwargs):
        """Log task completion with structured data."""
        self.info(
            "Task completed",
            task_id=task_id,
            success=success,
            execution_time=execution_time,
            **kwargs
        )
    
    def log_api_call(self, api_name: str, method: str, url: str, status_code: int = None, **kwargs):
        """Log API call with structured data."""
        self.info(
            "API call",
            api_name=api_name,
            method=method,
            url=url,
            status_code=status_code,
            **kwargs
        )