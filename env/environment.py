import json
import random
from .models import *

random.seed(42)

class CustomerSupportEnv:

    def __init__(self, max_steps=15):
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        with open("data/tickets.json") as f:
            data = json.load(f)

        self.tickets = [Ticket(**t) for t in data]
        self.time_step = 0
        self.last_action = None
        return self._get_obs()

    def state(self):
        return {
            "tickets": [t.dict() for t in self.tickets],
            "time_step": self.time_step
        }

    def step(self, action: Action):
        reward = 0.0
        done = False

        ticket = next((t for t in self.tickets if t.id == action.ticket_id), None)

        if not ticket:
            return self._get_obs(), -1.0, False, {}

        # Repeated action penalty
        if self.last_action == action.action_type:
            reward -= 0.2
        self.last_action = action.action_type

        # Classification
        if action.action_type == "classify":
            if ticket.category.lower() in (action.content or "").lower():
                reward += 0.3
            else:
                reward -= 0.2

        # Response
        elif action.action_type == "respond":
            if action.content and len(action.content) > 15:
                reward += 0.5
            else:
                reward -= 0.3

        # Escalate
        elif action.action_type == "escalate":
            if ticket.urgency == "high":
                reward += 0.4
            reward -= 0.1  # cost

        # Close
        elif action.action_type == "close":
            ticket.resolved = True
            reward += 0.3

        # SLA + Mood drift
        ticket.sla_deadline -= 1
        if ticket.sla_deadline <= 0 and not ticket.resolved:
            reward -= 0.5
            ticket.sentiment = "angry"

        # Queue pressure
        pending = sum(not t.resolved for t in self.tickets)
        if pending > 3:
            reward -= 0.2

        self.time_step += 1

        if self.time_step >= self.max_steps or pending == 0:
            done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        current = sorted(self.tickets, key=lambda t: (t.urgency, t.sla_deadline))[0]

        return Observation(
            queue=self.tickets,
            current_ticket=current,
            time_step=self.time_step,
            pending_count=sum(not t.resolved for t in self.tickets)
        )