"""
Configuration settings for the CRM Agent system.
"""
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # HubSpot Configuration
    hubspot_api_key: str = Field(..., env="HUBSPOT_API_KEY")
    hubspot_base_url: str = Field(default="https://api.hubapi.com", env="HUBSPOT_BASE_URL")
    
    # Email Configuration
    smtp_server: str = Field(default="smtp.gmail.com", env="SMTP_SERVER")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    email_username: str = Field(..., env="EMAIL_USERNAME")
    email_password: str = Field(..., env="EMAIL_PASSWORD")
    from_email: str = Field(..., env="FROM_EMAIL")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./crm_agent.db", env="DATABASE_URL")
    
    # FastAPI Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()