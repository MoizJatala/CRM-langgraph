"""
HubSpot Agent for CRM operations using LangGraph and function calling.
"""
import time
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict

from src.config.settings import settings
from src.models.schemas import ContactData, DealData, AgentResponse
from src.utils.logger import AgentLogger
from src.utils.helpers import sanitize_hubspot_property, filter_none_values, RetryHelper


class HubSpotAgentState(TypedDict):
    """State for HubSpot agent."""
    messages: List[BaseMessage]
    task_type: str
    payload: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    execution_time: Optional[float]


class HubSpotAgent:
    """HubSpot Agent for CRM operations."""
    
    def __init__(self):
        self.logger = AgentLogger("HubSpotAgent")
        self.api_key = settings.hubspot_api_key
        self.base_url = settings.hubspot_base_url
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=settings.openai_api_key
        )
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow for HubSpot operations."""
        
        def analyze_task(state: HubSpotAgentState) -> HubSpotAgentState:
            """Analyze the task and determine the appropriate action."""
            # Safely get task_type and payload with defaults
            task_type = state.get("task_type", "unknown")
            payload = state.get("payload", {})
            
            # Ensure messages list exists
            if "messages" not in state:
                state["messages"] = []
            
            # Add analysis message
            analysis_message = HumanMessage(
                content=f"Analyzing HubSpot task: {task_type} with payload: {payload}"
            )
            state["messages"].append(analysis_message)
            
            return state
        
        def execute_hubspot_operation(state: HubSpotAgentState) -> HubSpotAgentState:
            """Execute the HubSpot operation based on task type."""
            start_time = time.time()
            
            try:
                # Safely get task_type and payload with defaults
                task_type = state.get("task_type", "create_contact")
                payload = state.get("payload", {})
                
                # Ensure messages list exists
                if "messages" not in state:
                    state["messages"] = []
                
                if task_type == "create_contact":
                    result = self._create_contact(payload)
                elif task_type == "update_contact":
                    result = self._update_contact(payload)
                elif task_type == "create_deal":
                    result = self._create_deal(payload)
                elif task_type == "update_deal":
                    result = self._update_deal(payload)
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")
                
                state["result"] = result
                state["error"] = None
                
                # Add success message
                success_message = AIMessage(
                    content=f"Successfully completed {task_type} operation"
                )
                state["messages"].append(success_message)
                
            except Exception as e:
                self.logger.error(f"HubSpot operation failed: {str(e)}")
                state["error"] = str(e)
                state["result"] = None
                
                # Ensure messages list exists
                if "messages" not in state:
                    state["messages"] = []
                
                # Add error message
                error_message = AIMessage(
                    content=f"Failed to complete {task_type} operation: {str(e)}"
                )
                state["messages"].append(error_message)
            
            finally:
                state["execution_time"] = time.time() - start_time
            
            return state
        
        # Create the graph
        workflow = StateGraph(HubSpotAgentState)
        
        # Add nodes
        workflow.add_node("analyze", analyze_task)
        workflow.add_node("execute", execute_hubspot_operation)
        
        # Add edges
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "execute")
        workflow.add_edge("execute", END)
        
        return workflow.compile()
    
    async def process_task(self, task_type: str, payload: Dict[str, Any]) -> AgentResponse:
        """Process a HubSpot task using the LangGraph workflow."""
        start_time = time.time()
        
        try:
            # Initialize state
            initial_state = HubSpotAgentState(
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
                agent_name="HubSpotAgent",
                success=final_state["error"] is None,
                data=final_state["result"],
                error=final_state["error"],
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Task processing failed: {str(e)}")
            
            return AgentResponse(
                agent_name="HubSpotAgent",
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time
            )
    
    def _create_contact(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new contact in HubSpot."""
        contact_data = ContactData(**payload)
        
        # Prepare HubSpot properties
        properties = {
            "email": contact_data.email,
            "firstname": contact_data.firstname,
            "lastname": contact_data.lastname,
            "phone": contact_data.phone,
            "company": contact_data.company,
            "website": contact_data.website,
            "jobtitle": contact_data.jobtitle
        }
        
        # Filter out None values and sanitize
        properties = {
            k: sanitize_hubspot_property(v) 
            for k, v in filter_none_values(properties).items()
        }
        
        # Make API call
        url = f"{self.base_url}/crm/v3/objects/contacts"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {"properties": properties}
        
        with httpx.Client() as client:
            response = client.post(url, json=data, headers=headers)
            
            self.logger.log_api_call(
                api_name="HubSpot",
                method="POST",
                url=url,
                status_code=response.status_code
            )
            
            if response.status_code == 201:
                result = response.json()
                self.logger.info(f"Contact created successfully: {result['id']}")
                return {
                    "contact_id": result["id"],
                    "email": contact_data.email,
                    "created_at": result.get("createdAt"),
                    "properties": result.get("properties", {})
                }
            else:
                error_msg = f"Failed to create contact: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
    
    def _update_contact(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing contact in HubSpot."""
        contact_email = payload.get("email")
        if not contact_email:
            raise ValueError("Email is required for contact update")
        
        # First, find the contact by email
        contact_id = self._find_contact_by_email(contact_email)
        if not contact_id:
            raise ValueError(f"Contact not found with email: {contact_email}")
        
        # Prepare update properties
        update_data = {k: v for k, v in payload.items() if k != "email" and v is not None}
        properties = {
            k: sanitize_hubspot_property(v) 
            for k, v in update_data.items()
        }
        
        # Make API call
        url = f"{self.base_url}/crm/v3/objects/contacts/{contact_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {"properties": properties}
        
        with httpx.Client() as client:
            response = client.patch(url, json=data, headers=headers)
            
            self.logger.log_api_call(
                api_name="HubSpot",
                method="PATCH",
                url=url,
                status_code=response.status_code
            )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"Contact updated successfully: {contact_id}")
                return {
                    "contact_id": contact_id,
                    "email": contact_email,
                    "updated_at": result.get("updatedAt"),
                    "properties": result.get("properties", {})
                }
            else:
                error_msg = f"Failed to update contact: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
    
    def _create_deal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new deal in HubSpot."""
        # Preprocess payload to handle field name mappings
        processed_payload = self._preprocess_deal_payload(payload)
        deal_data = DealData(**processed_payload)
        
        # Prepare HubSpot properties
        properties = {
            "dealname": deal_data.dealname,
            "amount": deal_data.amount,
            "dealstage": deal_data.dealstage,
            "closedate": deal_data.closedate.isoformat() if deal_data.closedate else None,
            "pipeline": deal_data.pipeline,
            "hubspot_owner_id": deal_data.hubspot_owner_id
        }
        
        # Filter out None values and sanitize
        properties = {
            k: sanitize_hubspot_property(v) 
            for k, v in filter_none_values(properties).items()
        }
        
        # Make API call
        url = f"{self.base_url}/crm/v3/objects/deals"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {"properties": properties}
        
        with httpx.Client() as client:
            response = client.post(url, json=data, headers=headers)
            
            self.logger.log_api_call(
                api_name="HubSpot",
                method="POST",
                url=url,
                status_code=response.status_code
            )
            
            if response.status_code == 201:
                result = response.json()
                deal_id = result["id"]
                
                # Associate with contact if provided
                if deal_data.associated_contact_email:
                    self._associate_deal_with_contact(deal_id, deal_data.associated_contact_email)
                
                self.logger.info(f"Deal created successfully: {deal_id}")
                return {
                    "deal_id": deal_id,
                    "dealname": deal_data.dealname,
                    "created_at": result.get("createdAt"),
                    "properties": result.get("properties", {})
                }
            else:
                error_msg = f"Failed to create deal: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
    
    def _preprocess_deal_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess deal payload to handle field name mappings."""
        processed = payload.copy()
        
        # Handle field name mappings
        field_mappings = {
            "deal_name": "dealname",
            "deal_value": "amount", 
            "contact_email": "associated_contact_email"
        }
        
        for old_field, new_field in field_mappings.items():
            if old_field in processed:
                processed[new_field] = processed.pop(old_field)
        
        # Handle amount field - convert string values like "$10000" to float
        if "amount" in processed and isinstance(processed["amount"], str):
            amount_str = processed["amount"].replace("$", "").replace(",", "")
            try:
                processed["amount"] = float(amount_str)
            except ValueError:
                self.logger.warning(f"Could not convert amount '{processed['amount']}' to float")
                processed["amount"] = None
        
        return processed
    
    def _update_deal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing deal in HubSpot."""
        deal_id = payload.get("deal_id")
        if not deal_id:
            raise ValueError("Deal ID is required for deal update")
        
        # Prepare update properties
        update_data = {k: v for k, v in payload.items() if k != "deal_id" and v is not None}
        properties = {
            k: sanitize_hubspot_property(v) 
            for k, v in update_data.items()
        }
        
        # Make API call
        url = f"{self.base_url}/crm/v3/objects/deals/{deal_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {"properties": properties}
        
        with httpx.Client() as client:
            response = client.patch(url, json=data, headers=headers)
            
            self.logger.log_api_call(
                api_name="HubSpot",
                method="PATCH",
                url=url,
                status_code=response.status_code
            )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"Deal updated successfully: {deal_id}")
                return {
                    "deal_id": deal_id,
                    "updated_at": result.get("updatedAt"),
                    "properties": result.get("properties", {})
                }
            else:
                error_msg = f"Failed to update deal: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
    
    def _find_contact_by_email(self, email: str) -> Optional[str]:
        """Find contact ID by email address."""
        url = f"{self.base_url}/crm/v3/objects/contacts/search"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "filterGroups": [{
                "filters": [{
                    "propertyName": "email",
                    "operator": "EQ",
                    "value": email
                }]
            }]
        }
        
        with httpx.Client() as client:
            response = client.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    return results[0]["id"]
            
            return None
    
    def _associate_deal_with_contact(self, deal_id: str, contact_email: str):
        """Associate a deal with a contact."""
        contact_id = self._find_contact_by_email(contact_email)
        if not contact_id:
            self.logger.warning(f"Contact not found for association: {contact_email}")
            return
        
        url = f"{self.base_url}/crm/v3/objects/deals/{deal_id}/associations/contacts/{contact_id}/deal_to_contact"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        with httpx.Client() as client:
            response = client.put(url, headers=headers)
            
            if response.status_code == 200:
                self.logger.info(f"Deal {deal_id} associated with contact {contact_id}")
            else:
                self.logger.warning(f"Failed to associate deal with contact: {response.status_code}")


# Function calling tools for the HubSpot agent
@tool
def create_hubspot_contact(email: str, firstname: str = None, lastname: str = None, 
                          phone: str = None, company: str = None) -> Dict[str, Any]:
    """Create a new contact in HubSpot CRM."""
    agent = HubSpotAgent()
    payload = {
        "email": email,
        "firstname": firstname,
        "lastname": lastname,
        "phone": phone,
        "company": company
    }
    return agent._create_contact(payload)


@tool
def update_hubspot_contact(email: str, **kwargs) -> Dict[str, Any]:
    """Update an existing contact in HubSpot CRM."""
    agent = HubSpotAgent()
    payload = {"email": email, **kwargs}
    return agent._update_contact(payload)


@tool
def create_hubspot_deal(dealname: str, amount: float = None, dealstage: str = None,
                       associated_contact_email: str = None) -> Dict[str, Any]:
    """Create a new deal in HubSpot CRM."""
    agent = HubSpotAgent()
    payload = {
        "dealname": dealname,
        "amount": amount,
        "dealstage": dealstage,
        "associated_contact_email": associated_contact_email
    }
    return agent._create_deal(payload)


# Create module-level graph export for LangGraph
_hubspot_agent_instance = HubSpotAgent()
hubspot_agent_graph = _hubspot_agent_instance.graph

# Create LangGraph API-compatible graph
def create_langgraph_api_compatible_graph() -> StateGraph:
    """Create a LangGraph API-compatible version of the HubSpot agent graph."""
    from typing_extensions import TypedDict
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langgraph.graph import StateGraph, END
    
    class APICompatibleState(TypedDict):
        messages: List[BaseMessage]
    
    def process_hubspot_request(state: APICompatibleState) -> APICompatibleState:
        """Process HubSpot request from LangGraph API."""
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
                request_content = last_message.content
            elif isinstance(last_message, dict):
                request_content = last_message.get('content', '')
            else:
                request_content = str(last_message)
            
            # Try to parse the request content as JSON for task_type and payload
            import json
            import re
            try:
                request_data = json.loads(request_content)
                task_type = request_data.get("task_type", "create_contact")
                payload = request_data.get("payload", {})
            except json.JSONDecodeError:
                # If not JSON, treat as a simple contact creation request
                task_type = "create_contact"
                
                # Extract email from the text using regex
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                emails = re.findall(email_pattern, request_content)
                
                if emails:
                    # Use the first email found
                    payload = {"email": emails[0]}
                elif "@" in request_content:
                    # Fallback: if @ is present but regex didn't match, try to extract manually
                    parts = request_content.split()
                    for part in parts:
                        if "@" in part and "." in part:
                            payload = {"email": part.strip()}
                            break
                    else:
                        payload = {"name": request_content}
                else:
                    # No email found, treat as name
                    payload = {"name": request_content}
            
            # Create HubSpot agent instance and process task
            hubspot_agent = HubSpotAgent()
            
            # Use synchronous version for API compatibility
            import asyncio
            try:
                result = asyncio.run(hubspot_agent.process_task(task_type, payload))
                
                if result.success:
                    response_content = f"HubSpot operation '{task_type}' completed successfully. Data: {result.data}"
                else:
                    response_content = f"HubSpot operation '{task_type}' failed: {result.error}"
                    
            except Exception as e:
                response_content = f"Failed to process HubSpot request: {str(e)}"
            
            state["messages"].append(AIMessage(content=response_content))
            
        except Exception as e:
            error_message = f"Error processing HubSpot request: {str(e)}"
            state["messages"].append(AIMessage(content=error_message))
        
        return state
    
    # Create simple workflow
    workflow = StateGraph(APICompatibleState)
    workflow.add_node("process", process_hubspot_request)
    workflow.set_entry_point("process")
    workflow.add_edge("process", END)
    
    return workflow.compile()

# Export LangGraph API-compatible graph
langgraph_api_hubspot_graph = create_langgraph_api_compatible_graph()