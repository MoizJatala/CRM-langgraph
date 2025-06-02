"""
Helper utilities for the CRM Agent system.
"""
import uuid
import time
import json
from datetime import datetime
from typing import Any, Dict, Optional, List
from functools import wraps


def generate_task_id() -> str:
    """Generate a unique task ID."""
    return f"task_{uuid.uuid4().hex[:12]}"


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"session_{uuid.uuid4().hex[:8]}"


def current_timestamp() -> datetime:
    """Get current timestamp."""
    return datetime.utcnow()


def format_timestamp(dt: datetime) -> str:
    """Format datetime to ISO string."""
    return dt.isoformat()


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely load JSON string with fallback."""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """Safely dump object to JSON string with fallback."""
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return default


def extract_email_from_text(text: str) -> Optional[str]:
    """Extract email address from text using simple pattern matching."""
    import re
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, text)
    return matches[0] if matches else None


def extract_phone_from_text(text: str) -> Optional[str]:
    """Extract phone number from text using simple pattern matching."""
    import re
    # Simple phone pattern - can be enhanced
    phone_pattern = r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
    matches = re.findall(phone_pattern, text)
    if matches:
        match = matches[0]
        return f"{match[0]}{match[1]}-{match[2]}-{match[3]}".strip('-')
    return None


def clean_text(text: str) -> str:
    """Clean and normalize text input."""
    if not text:
        return ""
    return text.strip().replace('\n', ' ').replace('\r', ' ')


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    return wrapper


async def async_timing_decorator(func):
    """Decorator to measure async function execution time."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    return wrapper


def validate_email(email: str) -> bool:
    """Validate email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """Validate phone number format."""
    import re
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', phone)
    # Check if it's a valid US phone number (10 or 11 digits)
    return len(digits_only) in [10, 11]


def sanitize_hubspot_property(value: Any) -> str:
    """Sanitize value for HubSpot property."""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value).strip()


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries, with later ones taking precedence."""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def filter_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys with None values from dictionary."""
    return {k: v for k, v in data.items() if v is not None}


class RetryHelper:
    """Helper class for implementing retry logic."""
    
    @staticmethod
    def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Calculate exponential backoff delay."""
        delay = base_delay * (2 ** attempt)
        return min(delay, max_delay)
    
    @staticmethod
    async def retry_async(func, max_attempts: int = 3, base_delay: float = 1.0):
        """Retry async function with exponential backoff."""
        import asyncio
        
        last_exception = None
        for attempt in range(max_attempts):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    delay = RetryHelper.exponential_backoff(attempt, base_delay)
                    await asyncio.sleep(delay)
                else:
                    raise last_exception