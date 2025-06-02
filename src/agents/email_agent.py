"""
Email Agent for sending notifications using LangGraph and function calling.
"""
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict

from src.config.settings import settings
from src.models.schemas import EmailData, AgentResponse
from src.utils.logger import AgentLogger


class EmailAgentState(TypedDict):
    """State for Email agent."""
    messages: List[BaseMessage]
    task_type: str
    payload: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    execution_time: Optional[float]


class EmailAgent:
    """Email Agent for sending notifications."""
    
    def __init__(self):
        self.logger = AgentLogger("EmailAgent")
        self.smtp_server = settings.smtp_server
        self.smtp_port = settings.smtp_port
        self.username = settings.email_username
        self.password = settings.email_password
        self.from_email = settings.from_email
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            api_key=settings.openai_api_key
        )
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow for email operations."""
        
        def analyze_email_task(state: EmailAgentState) -> EmailAgentState:
            """Analyze the email task and prepare content."""
            # Safely get task_type and payload with defaults
            task_type = state.get("task_type", "unknown")
            payload = state.get("payload", {})
            
            # Ensure messages list exists
            if "messages" not in state:
                state["messages"] = []
            
            # Add analysis message
            analysis_message = HumanMessage(
                content=f"Analyzing email task: {task_type} with payload: {payload}"
            )
            state["messages"].append(analysis_message)
            
            return state
        
        def generate_email_content(state: EmailAgentState) -> EmailAgentState:
            """Generate or enhance email content using LLM."""
            try:
                # Safely get payload with default
                payload = state.get("payload", {})
                
                # Ensure messages list exists
                if "messages" not in state:
                    state["messages"] = []
                
                # Check if this is a specific email type that needs custom content
                email_type = payload.get("email_type", "")
                
                # If content is already provided and not a special type, use it as is
                if "body" in payload and payload["body"] and not email_type:
                    state["payload"]["enhanced_body"] = payload["body"]
                elif "enhanced_body" in payload and payload["enhanced_body"] and not email_type:
                    # Keep existing enhanced_body if no special type
                    pass
                else:
                    # Generate content based on email type or context
                    if email_type == "welcome":
                        prompt = self._create_welcome_email_prompt(payload)
                        if "payload" not in state:
                            state["payload"] = {}
                        state["payload"]["subject"] = "Welcome! We're excited to have you on board"
                    else:
                        context = payload.get("context", {})
                        action_type = context.get("action_type", "operation")
                        prompt = self._create_email_prompt(action_type, context)
                    
                    # Use LLM to generate content
                    response = self.llm.invoke([HumanMessage(content=prompt)])
                    if "payload" not in state:
                        state["payload"] = {}
                    state["payload"]["enhanced_body"] = response.content
                
                # Add content generation message
                content_message = AIMessage(
                    content="Email content generated successfully"
                )
                state["messages"].append(content_message)
                
            except Exception as e:
                self.logger.error(f"Content generation failed: {str(e)}")
                
                # Ensure messages list exists
                if "messages" not in state:
                    state["messages"] = []
                
                error_message = AIMessage(
                    content=f"Failed to generate email content: {str(e)}"
                )
                state["messages"].append(error_message)
            
            return state
        
        def send_email(state: EmailAgentState) -> EmailAgentState:
            """Send the email notification."""
            start_time = time.time()
            
            try:
                # Safely get payload with default
                payload = state.get("payload", {})
                
                # Ensure messages list exists
                if "messages" not in state:
                    state["messages"] = []
                
                # Handle both 'to_email' and 'email' field names
                to_email = payload.get("to_email") or payload.get("email")
                if not to_email:
                    raise ValueError("No recipient email address found in payload (expected 'to_email' or 'email' field)")
                
                # Prepare email data
                email_data = {
                    "to_email": to_email,
                    "subject": payload.get("subject", "CRM Operation Notification"),
                    "body": payload.get("enhanced_body", payload.get("body", "")),
                    "html_body": payload.get("html_body")
                }
                
                result = self._send_email_notification(email_data)
                
                state["result"] = result
                state["error"] = None
                
                # Add success message
                success_message = AIMessage(
                    content=f"Email sent successfully to {email_data['to_email']}"
                )
                state["messages"].append(success_message)
                
            except Exception as e:
                self.logger.error(f"Email sending failed: {str(e)}")
                state["error"] = str(e)
                state["result"] = None
                
                # Ensure messages list exists
                if "messages" not in state:
                    state["messages"] = []
                
                # Add error message
                error_message = AIMessage(
                    content=f"Failed to send email: {str(e)}"
                )
                state["messages"].append(error_message)
            
            finally:
                state["execution_time"] = time.time() - start_time
            
            return state
        
        # Create the graph
        workflow = StateGraph(EmailAgentState)
        
        # Add nodes
        workflow.add_node("analyze", analyze_email_task)
        workflow.add_node("generate_content", generate_email_content)
        workflow.add_node("send", send_email)
        
        # Add edges
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "generate_content")
        workflow.add_edge("generate_content", "send")
        workflow.add_edge("send", END)
        
        return workflow.compile()
    
    async def process_task(self, task_type: str, payload: Dict[str, Any]) -> AgentResponse:
        """Process an email task using the LangGraph workflow."""
        start_time = time.time()
        
        try:
            # Initialize state
            initial_state = EmailAgentState(
                messages=[],
                task_type=task_type,
                payload=payload,
                result=None,
                error=None,
                execution_time=None
            )
            
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                agent_name="EmailAgent",
                success=final_state["error"] is None,
                data=final_state["result"],
                error=final_state["error"],
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Task processing failed: {str(e)}")
            
            return AgentResponse(
                agent_name="EmailAgent",
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time
            )
    
    def _create_email_prompt(self, action_type: str, context: Dict[str, Any]) -> str:
        """Create a prompt for email content generation."""
        base_prompt = f"""
        Generate a professional email notification for a CRM operation.
        
        Action Type: {action_type}
        Context: {context}
        
        The email should:
        1. Be professional and concise
        2. Clearly state what action was performed
        3. Include relevant details from the context
        4. Have a friendly but business-appropriate tone
        5. Be around 100-150 words
        
        Generate only the email body content, no subject line.
        """
        
        return base_prompt.strip()
    
    def _create_welcome_email_prompt(self, payload: Dict[str, Any]) -> str:
        """Create a prompt for welcome email content generation."""
        recipient_email = payload.get("email") or payload.get("to_email", "valued customer")
        
        welcome_prompt = f"""
        Generate a warm and professional welcome email for a new user or contact.
        
        Recipient: {recipient_email}
        
        The welcome email should:
        1. Be warm, friendly, and welcoming
        2. Express excitement about having them join
        3. Briefly mention what they can expect from your service/platform
        4. Include a call to action (like exploring features, contacting support, etc.)
        5. Be professional but personable
        6. Be around 100-150 words
        
        Generate only the email body content, no subject line.
        """
        
        return welcome_prompt.strip()
    
    def _send_email_notification(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send email notification via SMTP."""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = email_data["subject"]
            msg['From'] = self.from_email
            msg['To'] = email_data["to_email"]
            
            # Create text part
            text_part = MIMEText(email_data["body"], 'plain')
            msg.attach(text_part)
            
            # Create HTML part if provided
            if email_data.get("html_body"):
                html_part = MIMEText(email_data["html_body"], 'html')
                msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            self.logger.info(f"Email sent successfully to {email_data['to_email']}")
            
            return {
                "status": "sent",
                "to_email": email_data["to_email"],
                "subject": email_data["subject"],
                "sent_at": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {str(e)}")
            raise
    
    def create_confirmation_email(self, operation_type: str, operation_data: Dict[str, Any], 
                                recipient_email: str) -> Dict[str, Any]:
        """Create a confirmation email for CRM operations."""
        
        # Generate subject based on operation type
        subject_map = {
            "create_contact": "Contact Created Successfully",
            "update_contact": "Contact Updated Successfully", 
            "create_deal": "Deal Created Successfully",
            "update_deal": "Deal Updated Successfully"
        }
        
        subject = subject_map.get(operation_type, "CRM Operation Completed")
        
        # Create context for email generation
        context = {
            "action_type": operation_type,
            "operation_data": operation_data,
            "timestamp": time.time()
        }
        
        return {
            "to_email": recipient_email,
            "subject": subject,
            "context": context
        }


# Function calling tools for the Email agent
@tool
def send_email_notification(to_email: str, subject: str, body: str, 
                          html_body: str = None) -> Dict[str, Any]:
    """Send an email notification."""
    agent = EmailAgent()
    email_data = {
        "to_email": to_email,
        "subject": subject,
        "body": body,
        "html_body": html_body
    }
    return agent._send_email_notification(email_data)


@tool
def send_crm_confirmation_email(operation_type: str, operation_data: dict, 
                              recipient_email: str) -> Dict[str, Any]:
    """Send a confirmation email for CRM operations."""
    agent = EmailAgent()
    email_payload = agent.create_confirmation_email(
        operation_type, operation_data, recipient_email
    )
    return agent._send_email_notification(email_payload)


@tool
def generate_email_content(action_type: str, context: dict) -> str:
    """Generate email content using AI."""
    agent = EmailAgent()
    prompt = agent._create_email_prompt(action_type, context)
    
    response = agent.llm.invoke([HumanMessage(content=prompt)])
    return response.content


# Create module-level graph export for LangGraph
_email_agent_instance = EmailAgent()
email_agent_graph = _email_agent_instance.graph

# Create LangGraph API-compatible graph
def create_langgraph_api_compatible_graph() -> StateGraph:
    """Create a LangGraph API-compatible version of the email agent graph."""
    from typing_extensions import TypedDict
    from langchain_core.messages import BaseMessage
    
    class APICompatibleState(TypedDict):
        messages: List[BaseMessage]
    
    def process_email_request(state: APICompatibleState) -> APICompatibleState:
        """Process email request from LangGraph API."""
        try:
            # Get the last human/user message - handle both LangChain and API formats
            human_messages = []
            for msg in state["messages"]:
                # Check for LangChain format (type='human') or API format (role='user')
                if (hasattr(msg, 'type') and msg.type == 'human') or \
                   (hasattr(msg, 'role') and msg.role == 'user') or \
                   (isinstance(msg, dict) and msg.get('role') == 'user'):
                    human_messages.append(msg)
            
            if not human_messages:
                state["messages"].append(AIMessage(content="No human message found to process."))
                return state
            
            last_message = human_messages[-1]
            
            # Extract content from message (handle both formats)
            if hasattr(last_message, 'content'):
                content = last_message.content
            elif isinstance(last_message, dict):
                content = last_message.get('content', '')
            else:
                content = str(last_message)
            
            # Try to parse the content as a task request
            import json
            try:
                # Attempt to parse as JSON
                task_data = json.loads(content)
                task_type = task_data.get("task_type", "send_email")
                payload = task_data.get("payload", {})
            except json.JSONDecodeError:
                # Fallback: treat as a simple email request
                task_type = "send_email"
                payload = {
                    "to_email": "recipient@example.com",
                    "subject": "Notification",
                    "body": content
                }
            
            # Create agent instance and process task
            agent = EmailAgent()
            
            # Use synchronous version for API compatibility
            import asyncio
            try:
                result = asyncio.run(agent.process_task(task_type, payload))
                
                if result.success:
                    response_content = f"Email task completed successfully. Data: {result.data}"
                else:
                    response_content = f"Email task failed: {result.error}"
                    
            except Exception as e:
                response_content = f"Failed to process email task: {str(e)}"
            
            state["messages"].append(AIMessage(content=response_content))
            
        except Exception as e:
            error_message = f"Error processing email request: {str(e)}"
            state["messages"].append(AIMessage(content=error_message))
        
        return state
    
    # Create simple workflow
    workflow = StateGraph(APICompatibleState)
    workflow.add_node("process", process_email_request)
    workflow.set_entry_point("process")
    workflow.add_edge("process", END)
    
    return workflow.compile()

# Export LangGraph API-compatible graph
langgraph_api_email_graph = create_langgraph_api_compatible_graph()