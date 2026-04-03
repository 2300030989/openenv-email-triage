from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class ActionType(str, Enum):
    ARCHIVE = "archive"
    REPLY = "reply"
    FORWARD = "forward"
    MARK_URGENT = "mark_urgent"
    CREATE_CALENDAR_EVENT = "create_calendar_event"
    WAIT = "wait"

class Action(BaseModel):
    action_type: ActionType
    email_id: Optional[str] = None
    content: Optional[str] = None
    recipient: Optional[str] = None
    event_details: Optional[Dict[str, Any]] = None

class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    is_urgent: bool = False
    is_read: bool = False
    thread_id: Optional[str] = None

class Observation(BaseModel):
    inbox: List[Email]
    current_email: Optional[Email] = None
    calendar_events: List[Dict[str, Any]] = []
    unread_count: int
    last_action_status: str = "idle"

class Reward(BaseModel):
    value: float = Field(..., ge=-10.0, le=10.0)
    reason: str
    partial_progress: float = 0.0
