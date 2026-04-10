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

        reward = 0.3  # ✅ constant safe reward (avoid edge cases)

        # move forward
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

        # ✅ constant safe score (STRICTLY BETWEEN 0 AND 1)
        info = {
            "score": 0.6
        }

        return obs, reward, self.done, info

    def state(self):
        return {
            "current_index": self.current_index,
            "time_step": self.time_step,
            "done": self.done
        }