"""
Global Orchestrator Agent for coordinating multi-agent CRM operations using LangGraph.
"""
import time
import asyncio
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict

from src.config.settings import settings
from src.models.schemas import (
    UserQuery, TaskRequest, TaskType, AgentResponse, OrchestrationResult
)
from src.agents.hubspot_agent import HubSpotAgent
from src.agents.email_agent import EmailAgent
from src.utils.logger import AgentLogger
from src.utils.helpers import generate_task_id, extract_email_from_text, clean_text


class OrchestratorState(TypedDict):
    """State for Orchestrator agent."""
    messages: List[BaseMessage]
    user_query: str
    identified_tasks: List[TaskRequest]
    hubspot_results: List[AgentResponse]
    email_results: List[AgentResponse]
    overall_success: bool
    execution_time: Optional[float]
    summary: str


class GlobalOrchestratorAgent:
    """Global Orchestrator Agent for coordinating CRM operations."""
    
    def __init__(self):
        self.logger = AgentLogger("GlobalOrchestrator")
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        self.hubspot_agent = HubSpotAgent()
        self.email_agent = EmailAgent()
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow for orchestration."""
        
        def analyze_user_query(state: OrchestratorState) -> OrchestratorState:
            """Analyze user query and identify required tasks."""
            # Safely get user_query with default
            user_query = state.get("user_query", "")
            
            # Ensure messages list exists
            if "messages" not in state:
                state["messages"] = []
            
            # Create system prompt for task identification
            system_prompt = """
            You are a CRM operations analyst. Analyze the user query and identify what CRM operations need to be performed.
            
            Available operations:
            1. create_contact - Create a new contact
            2. update_contact - Update existing contact
            3. create_deal - Create a new deal
            4. update_deal - Update existing deal
            5. send_email - Send email notification
            
            For each operation, extract the relevant data from the user query.
            
            Respond in JSON format with a list of tasks:
            {
                "tasks": [
                    {
                        "task_type": "operation_type",
                        "payload": {"key": "value"},
                        "priority": 1
                    }
                ]
            }
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User query: {user_query}")
            ]
            
            try:
                response = self.llm.invoke(messages)
                
                # Parse the response and create task requests
                tasks = self._parse_task_response(response.content, user_query)
                state["identified_tasks"] = tasks
                
                # Add analysis message
                analysis_message = AIMessage(
                    content=f"Identified {len(tasks)} tasks from user query"
                )
                state["messages"].append(analysis_message)
                
            except Exception as e:
                self.logger.error(f"Query analysis failed: {str(e)}")
                # Fallback: try to extract basic information
                tasks = self._fallback_task_extraction(user_query)
                state["identified_tasks"] = tasks
                
                error_message = AIMessage(
                    content=f"Used fallback analysis, identified {len(tasks)} tasks"
                )
                state["messages"].append(error_message)
            
            return state
        
        async def execute_hubspot_tasks(state: OrchestratorState) -> OrchestratorState:
            """Execute HubSpot-related tasks."""
            # Safely get identified_tasks with default
            identified_tasks = state.get("identified_tasks", [])
            
            hubspot_tasks = [
                task for task in identified_tasks
                if task.task_type in [TaskType.CREATE_CONTACT, TaskType.UPDATE_CONTACT, 
                                    TaskType.CREATE_DEAL, TaskType.UPDATE_DEAL]
            ]
            
            # Ensure messages list exists
            if "messages" not in state:
                state["messages"] = []
            
            hubspot_results = []
            
            for task in hubspot_tasks:
                try:
                    # Preprocess payload for contact operations
                    processed_payload = self._preprocess_payload(task.task_type, task.payload)
                    
                    # Execute task asynchronously and await the result
                    result = await self.hubspot_agent.process_task(task.task_type.value, processed_payload)
                    
                    hubspot_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"HubSpot task execution failed: {str(e)}")
                    error_result = AgentResponse(
                        agent_name="HubSpotAgent",
                        success=False,
                        data=None,
                        error=str(e),
                        execution_time=0.0
                    )
                    hubspot_results.append(error_result)
            
            state["hubspot_results"] = hubspot_results
            
            # Add execution message
            execution_message = AIMessage(
                content=f"Executed {len(hubspot_tasks)} HubSpot tasks"
            )
            state["messages"].append(execution_message)
            
            return state
        
        async def execute_email_tasks(state: OrchestratorState) -> OrchestratorState:
            """Execute email-related tasks and send confirmations."""
            # Safely get identified_tasks and hubspot_results with defaults
            identified_tasks = state.get("identified_tasks", [])
            hubspot_results = state.get("hubspot_results", [])
            user_query = state.get("user_query", "")
            
            # Ensure messages list exists
            if "messages" not in state:
                state["messages"] = []
            
            # Get explicit email tasks
            email_tasks = [
                task for task in identified_tasks
                if task.task_type == TaskType.SEND_EMAIL
            ]
            
            # Create confirmation emails for successful HubSpot operations
            confirmation_tasks = []
            for result in hubspot_results:
                if hasattr(result, 'success') and result.success and result.data:
                    # Extract recipient email from the operation data
                    recipient_email = self._extract_recipient_email(result.data, user_query)
                    if recipient_email:
                        confirmation_task = self._create_confirmation_task(result, recipient_email)
                        confirmation_tasks.append(confirmation_task)
            
            all_email_tasks = email_tasks + confirmation_tasks
            email_results = []
            
            for task in all_email_tasks:
                try:
                    result = await self.email_agent.process_task(task.task_type.value, task.payload)
                    
                    email_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Email task execution failed: {str(e)}")
                    error_result = AgentResponse(
                        agent_name="EmailAgent",
                        success=False,
                        data=None,
                        error=str(e),
                        execution_time=0.0
                    )
                    email_results.append(error_result)
            
            state["email_results"] = email_results
            
            # Add execution message
            execution_message = AIMessage(
                content=f"Executed {len(all_email_tasks)} email tasks"
            )
            state["messages"].append(execution_message)
            
            return state
        
        def generate_summary(state: OrchestratorState) -> OrchestratorState:
            """Generate a summary of the orchestration results."""
            # Safely get results with defaults
            hubspot_results = state.get("hubspot_results", [])
            email_results = state.get("email_results", [])
            user_query = state.get("user_query", "")
            
            # Ensure messages list exists
            if "messages" not in state:
                state["messages"] = []
            
            all_results = hubspot_results + email_results
            
            successful_operations = sum(1 for result in all_results if hasattr(result, 'success') and result.success)
            total_operations = len(all_results)
            
            state["overall_success"] = successful_operations == total_operations and total_operations > 0
            
            # Generate summary using LLM
            summary_prompt = f"""
            Generate a concise summary of the CRM operations performed:
            
            User Query: {user_query}
            Total Operations: {total_operations}
            Successful Operations: {successful_operations}
            
            Operations Details:
            {self._format_results_for_summary(all_results)}
            
            Provide a brief, user-friendly summary of what was accomplished.
            """
            
            try:
                summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
                state["summary"] = summary_response.content
            except Exception as e:
                self.logger.error(f"Summary generation failed: {str(e)}")
                state["summary"] = f"Completed {successful_operations}/{total_operations} operations successfully."
            
            # Add summary message
            summary_message = AIMessage(content=state["summary"])
            state["messages"].append(summary_message)
            
            return state
        
        # Create the graph
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes
        workflow.add_node("analyze", analyze_user_query)
        workflow.add_node("execute_hubspot", execute_hubspot_tasks)
        workflow.add_node("execute_email", execute_email_tasks)
        workflow.add_node("summarize", generate_summary)
        
        # Add edges
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "execute_hubspot")
        workflow.add_edge("execute_hubspot", "execute_email")
        workflow.add_edge("execute_email", "summarize")
        workflow.add_edge("summarize", END)
        
        return workflow.compile()
    
    async def process_user_query(self, user_query: str) -> OrchestrationResult:
        """Process a user query through the orchestration workflow."""
        start_time = time.time()
        task_id = generate_task_id()
        
        try:
            # Initialize state
            initial_state = OrchestratorState(
                messages=[],
                user_query=clean_text(user_query),
                identified_tasks=[],
                hubspot_results=[],
                email_results=[],
                overall_success=False,
                execution_time=None,
                summary=""
            )
            
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            execution_time = time.time() - start_time
            final_state["execution_time"] = execution_time
            
            # Collect all results
            all_results = final_state["hubspot_results"] + final_state["email_results"]
            
            return OrchestrationResult(
                task_id=task_id,
                user_query=user_query,
                identified_tasks=final_state["identified_tasks"],
                results=all_results,
                overall_success=final_state["overall_success"],
                total_execution_time=execution_time,
                summary=final_state["summary"]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Orchestration failed: {str(e)}")
            
            return OrchestrationResult(
                task_id=task_id,
                user_query=user_query,
                identified_tasks=[],
                results=[],
                overall_success=False,
                total_execution_time=execution_time,
                summary=f"Orchestration failed: {str(e)}"
            )
    
    def _parse_task_response(self, response_content: str, user_query: str) -> List[TaskRequest]:
        """Parse LLM response to extract task requests."""
        import json
        
        try:
            # Try to parse JSON response
            response_data = json.loads(response_content)
            tasks = []
            
            for task_data in response_data.get("tasks", []):
                task = TaskRequest(
                    task_type=TaskType(task_data["task_type"]),
                    payload=task_data["payload"],
                    user_query=user_query,
                    priority=task_data.get("priority", 1)
                )
                tasks.append(task)
            
            return tasks
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Failed to parse LLM response: {str(e)}")
            return self._fallback_task_extraction(user_query)
    
    def _fallback_task_extraction(self, user_query: str) -> List[TaskRequest]:
        """Fallback method to extract tasks from user query using simple heuristics."""
        tasks = []
        query_lower = user_query.lower()
        
        # Extract email if present
        email = extract_email_from_text(user_query)
        
        # Simple keyword-based task identification
        if "create contact" in query_lower or "new contact" in query_lower:
            payload = {"email": email} if email else {}
            tasks.append(TaskRequest(
                task_type=TaskType.CREATE_CONTACT,
                payload=payload,
                user_query=user_query,
                priority=1
            ))
        
        if "create deal" in query_lower or "new deal" in query_lower:
            payload = {"dealname": "New Deal", "associated_contact_email": email} if email else {"dealname": "New Deal"}
            tasks.append(TaskRequest(
                task_type=TaskType.CREATE_DEAL,
                payload=payload,
                user_query=user_query,
                priority=1
            ))
        
        if "send email" in query_lower or "email notification" in query_lower:
            if email:
                payload = {
                    "to_email": email,
                    "subject": "CRM Notification",
                    "body": "This is a notification from your CRM system."
                }
                tasks.append(TaskRequest(
                    task_type=TaskType.SEND_EMAIL,
                    payload=payload,
                    user_query=user_query,
                    priority=2
                ))
        
        return tasks
    
    def _extract_recipient_email(self, operation_data: Dict[str, Any], user_query: str) -> Optional[str]:
        """Extract recipient email for confirmation emails."""
        # Try to get email from operation data
        if "email" in operation_data:
            return operation_data["email"]
        
        # Try to extract from user query
        return extract_email_from_text(user_query)
    
    def _create_confirmation_task(self, operation_result: AgentResponse, recipient_email: str) -> TaskRequest:
        """Create a confirmation email task for a successful operation."""
        # Determine operation type from agent response
        operation_type = "operation"
        if operation_result.data:
            if "contact_id" in operation_result.data:
                operation_type = "create_contact" if "created_at" in operation_result.data else "update_contact"
            elif "deal_id" in operation_result.data:
                operation_type = "create_deal" if "created_at" in operation_result.data else "update_deal"
        
        payload = {
            "to_email": recipient_email,
            "subject": f"CRM Operation Completed: {operation_type.replace('_', ' ').title()}",
            "context": {
                "action_type": operation_type,
                "operation_data": operation_result.data,
                "timestamp": time.time()
            }
        }
        
        return TaskRequest(
            task_type=TaskType.SEND_EMAIL,
            payload=payload,
            user_query="Confirmation email",
            priority=3
        )
    
    def _format_results_for_summary(self, results: List[AgentResponse]) -> str:
        """Format results for summary generation."""
        formatted_results = []
        
        for result in results:
            status = "Success" if hasattr(result, 'success') and result.success else "Failed"
            agent = getattr(result, 'agent_name', 'Unknown')
            error = getattr(result, 'error', None)
            
            result_str = f"- {agent}: {status}"
            if error:
                result_str += f" (Error: {error})"
            
            formatted_results.append(result_str)
        
        return "\n".join(formatted_results)

    def _preprocess_payload(self, task_type: TaskType, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess payload for contact operations."""
        if task_type in [TaskType.CREATE_CONTACT, TaskType.UPDATE_CONTACT]:
            # Map "name" to "firstname" and "lastname"
            if "name" in payload:
                name = payload["name"]
                if isinstance(name, str):
                    parts = name.split()
                    if len(parts) == 1:
                        payload["firstname"] = parts[0]
                    elif len(parts) >= 2:
                        payload["firstname"] = parts[0]
                        payload["lastname"] = " ".join(parts[1:])
                    # Remove the original "name" field
                    del payload["name"]
                elif isinstance(name, dict):
                    payload["firstname"] = name.get("firstname")
                    payload["lastname"] = name.get("lastname")
                    # Remove the original "name" field
                    del payload["name"]
        return payload


# Function calling tools for the Orchestrator
@tool
def process_crm_query(user_query: str) -> Dict[str, Any]:
    """Process a user query for CRM operations."""
    orchestrator = GlobalOrchestratorAgent()
    result = asyncio.run(orchestrator.process_user_query(user_query))
    
    return {
        "task_id": result.task_id,
        "success": result.overall_success,
        "summary": result.summary,
        "execution_time": result.total_execution_time,
        "tasks_completed": len(result.results)
    }


@tool
def analyze_user_intent(user_query: str) -> Dict[str, Any]:
    """Analyze user intent and identify required CRM operations."""
    orchestrator = GlobalOrchestratorAgent()
    
    # Use the LLM to analyze intent
    system_prompt = """
    Analyze the user query and identify the intent and required CRM operations.
    Return a structured analysis including:
    1. Primary intent
    2. Required operations
    3. Extracted entities (emails, names, etc.)
    4. Confidence level
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]
    
    response = orchestrator.llm.invoke(messages)
    
    return {
        "user_query": user_query,
        "analysis": response.content,
        "extracted_email": extract_email_from_text(user_query)
    }


# Create module-level graph export for LangGraph
_orchestrator_agent_instance = GlobalOrchestratorAgent()
orchestrator_agent_graph = _orchestrator_agent_instance.graph

# Create LangGraph API-compatible graph
def create_langgraph_api_compatible_graph() -> StateGraph:
    """Create a LangGraph API-compatible version of the orchestrator agent graph."""
    from typing_extensions import TypedDict
    from langchain_core.messages import BaseMessage
    
    class APICompatibleState(TypedDict):
        messages: List[BaseMessage]
    
    def process_orchestrator_request(state: APICompatibleState) -> APICompatibleState:
        """Process orchestrator request from LangGraph API."""
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
                user_query = last_message.content
            elif isinstance(last_message, dict):
                user_query = last_message.get('content', '')
            else:
                user_query = str(last_message)
            
            # Create orchestrator instance and process query
            orchestrator = GlobalOrchestratorAgent()
            
            # Use synchronous version for API compatibility
            import asyncio
            try:
                result = asyncio.run(orchestrator.process_user_query(user_query))
                
                if result.overall_success:
                    response_content = f"Orchestration completed successfully. Summary: {result.summary}"
                else:
                    response_content = f"Orchestration completed with issues. Summary: {result.summary}"
                    
            except Exception as e:
                response_content = f"Failed to process orchestration request: {str(e)}"
            
            state["messages"].append(AIMessage(content=response_content))
            
        except Exception as e:
            error_message = f"Error processing orchestrator request: {str(e)}"
            state["messages"].append(AIMessage(content=error_message))
        
        return state
    
    # Create simple workflow
    workflow = StateGraph(APICompatibleState)
    workflow.add_node("process", process_orchestrator_request)
    workflow.set_entry_point("process")
    workflow.add_edge("process", END)
    
    return workflow.compile()

# Export LangGraph API-compatible graph
langgraph_api_orchestrator_graph = create_langgraph_api_compatible_graph()