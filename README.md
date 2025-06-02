# CRM Agent System

A multi-agent AI system for CRM operations using LangGraph, designed to automate and streamline customer relationship management tasks through intelligent agent coordination.

## 🚀 Features

- **Multi-Agent Architecture**: Coordinated agents for different CRM operations
- **HubSpot Integration**: Create and update contacts, deals, and manage CRM data
- **Email Automation**: Intelligent email notifications and confirmations
- **Natural Language Processing**: Process user queries in natural language
- **RESTful API**: FastAPI-based web service with comprehensive endpoints
- **Database Management**: SQLite with Alembic migrations
- **Structured Logging**: Comprehensive logging with structured output
- **Error Handling**: Robust error handling and retry mechanisms

## 🏗️ Architecture

### Agents

1. **Global Orchestrator Agent**: Coordinates operations between specialized agents
2. **HubSpot Agent**: Handles CRM operations (contacts, deals)
3. **Email Agent**: Manages email notifications and content generation

### Technology Stack

- **Framework**: LangGraph for agent workflows
- **API**: FastAPI for web service
- **Database**: SQLite with SQLAlchemy ORM
- **Migrations**: Alembic for database schema management
- **AI/ML**: OpenAI GPT-4 for natural language processing
- **Validation**: Pydantic for data validation
- **Logging**: Structlog for structured logging

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- HubSpot API key
- Email account with app password (for SMTP)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Hubspot-CRM
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your configuration:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   HUBSPOT_API_KEY=your_hubspot_api_key_here
   EMAIL_USERNAME=your_email@gmail.com
   EMAIL_PASSWORD=your_app_password_here
   ```

5. **Initialize database**:
   ```bash
   alembic upgrade head
   ```

## 🚀 Usage

### Starting the Application

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

### Example API Calls

#### Process Natural Language Query
```bash
curl -X POST "http://localhost:8000/process-query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Create a new contact for john.doe@example.com with name John Doe and send a welcome email"
     }'
```

#### Create HubSpot Contact
```bash
curl -X POST "http://localhost:8000/hubspot/create-contact" \
     -H "Content-Type: application/json" \
     -d '{
       "email": "john.doe@example.com",
       "firstname": "John",
       "lastname": "Doe",
       "company": "Example Corp"
     }'
```

#### Send Email Notification
```bash
curl -X POST "http://localhost:8000/email/send-notification" \
     -H "Content-Type: application/json" \
     -d '{
       "to_email": "john.doe@example.com",
       "subject": "Welcome to our CRM",
       "body": "Thank you for joining our system!"
     }'
```

## 📁 Project Structure

```
Hubspot-CRM/
├── src/
│   ├── agents/              # Agent implementations
│   │   ├── orchestrator_agent.py
│   │   ├── hubspot_agent.py
│   │   └── email_agent.py
│   ├── api/                 # FastAPI application
│   │   └── main.py
│   ├── config/              # Configuration management
│   │   └── settings.py
│   ├── database/            # Database models and connection
│   │   ├── connection.py
│   │   └── models.py
│   ├── models/              # Pydantic schemas
│   │   └── schemas.py
│   └── utils/               # Utility functions
│       ├── logger.py
│       └── helpers.py
├── alembic/                 # Database migrations
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
├── alembic.ini             # Alembic configuration
└── main.py                 # Application entry point
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `HUBSPOT_API_KEY` | HubSpot API key | Required |
| `HUBSPOT_BASE_URL` | HubSpot API base URL | `https://api.hubapi.com` |
| `SMTP_SERVER` | SMTP server for email | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP port | `587` |
| `EMAIL_USERNAME` | Email username | Required |
| `EMAIL_PASSWORD` | Email password/app password | Required |
| `DATABASE_URL` | Database connection URL | `sqlite:///./crm_agent.db` |
| `API_HOST` | API host | `0.0.0.0` |
| `API_PORT` | API port | `8000` |
| `DEBUG` | Debug mode | `True` |
| `LOG_LEVEL` | Logging level | `INFO` |

## 🗄️ Database

The system uses SQLite by default with the following tables:

- **task_history**: Stores task execution history
- **contact_cache**: Caches HubSpot contact data
- **deal_cache**: Caches HubSpot deal data
- **agent_execution**: Tracks individual agent executions

### Database Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## 🧪 Testing

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_agents.py
```

## 📊 Monitoring and Logging

The system uses structured logging with the following features:

- **Agent-specific logging**: Each agent has its own logger
- **Task tracking**: All tasks are logged with execution times
- **API call logging**: External API calls are logged
- **Error tracking**: Comprehensive error logging with context

Logs are output in JSON format for easy parsing and monitoring.

## 🔒 Security Considerations

- Store API keys securely using environment variables
- Use app passwords for email authentication
- Configure CORS appropriately for production
- Implement rate limiting for production deployments
- Use HTTPS in production environments

## 🚀 Deployment

### Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Production Considerations

- Use a production WSGI server (e.g., Gunicorn)
- Set up proper logging aggregation
- Configure environment-specific settings
- Implement health checks and monitoring
- Use a production database (PostgreSQL, MySQL)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

1. Check the [API documentation](http://localhost:8000/docs)
2. Review the logs for error details
3. Create an issue in the repository

## 🔄 Changelog

### Version 1.0.0
- Initial release
- Multi-agent architecture implementation
- HubSpot and Email agent integration
- FastAPI web service
- Database management with Alembic
- Comprehensive logging and error handling