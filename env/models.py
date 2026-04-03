from pydantic import BaseModel
from typing import List, Optional

class Ticket(BaseModel):
    id: int
    message: str
    sentiment: str
    urgency: str
    category: str
    sla_deadline: int
    customer_history: int
    resolved: bool = False

class Observation(BaseModel):
    queue: List[Ticket]
    current_ticket: Optional[Ticket]
    time_step: int
    pending_count: int

class Action(BaseModel):
    action_type: str
    ticket_id: int
    content: Optional[str] = None