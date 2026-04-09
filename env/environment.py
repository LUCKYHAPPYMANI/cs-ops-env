import random
from env.models import Observation, Ticket


class CustomerSupportEnv:
    def __init__(self):
        self.tickets = []
        self.current_index = 0
        self.done = False
        self.time_step = 0

    def reset(self):
        random.seed(42)

        self.current_index = 0
        self.done = False
        self.time_step = 0

        self.tickets = [
            Ticket(
                id=1,
                message="I want a refund",
                urgency="high",
                sentiment="angry",
                sla_deadline=2,
                resolved=False,
                category="billing",
                customer_history=5
            ),
            Ticket(
                id=2,
                message="Order not delivered",
                urgency="medium",
                sentiment="neutral",
                sla_deadline=3,
                resolved=False,
                category="delivery",
                customer_history=1
            ),
            Ticket(
                id=3,
                message="App not working",
                urgency="low",
                sentiment="angry",
                sla_deadline=1,
                resolved=False,
                category="technical",
                customer_history=3
            )
        ]

        return Observation(
            current_ticket=self.tickets[self.current_index],
            queue=self.tickets,
            time_step=self.time_step,
            pending_count=len(self.tickets)
        )

    def step(self, action):
        ticket = self.tickets[self.current_index]

        reward = 0.0

        # ✅ Reward logic
        if action.action_type == "escalate" and ticket.urgency == "high":
            reward += 0.5
        elif action.action_type == "classify" and "refund" in ticket.message.lower():
            reward += 0.4
        elif action.action_type == "respond":
            reward += 0.3
        elif action.action_type == "close" and ticket.sla_deadline <= 1:
            reward += 0.5
            ticket.resolved = True
        else:
            reward -= 0.2

        # update state
        self.current_index += 1
        self.time_step += 1

        if self.current_index >= len(self.tickets):
            self.done = True
            next_ticket = ticket
        else:
            next_ticket = self.tickets[self.current_index]

        obs = Observation(
            current_ticket=next_ticket,
            queue=self.tickets,
            time_step=self.time_step,
            pending_count=len(self.tickets) - self.current_index
        )

        # 🔥 FINAL SCORE FIX (ABSOLUTE GUARANTEE)
        raw_score = float(reward)

        # Normalize into safe range
        score = 0.5 + (raw_score * 0.4)

        # Clamp strictly between (0,1)
        if score <= 0:
            score = 0.01
        elif score >= 1:
            score = 0.99

        info = {
            "score": score
        }

        return obs, reward, self.done, info

    def state(self):
        return {
            "current_index": self.current_index,
            "time_step": self.time_step,
            "done": self.done
        }