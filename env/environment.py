from env.models import Observation, Ticket

print("LOADED REAL environment.py")


class CustomerSupportEnv:
    def __init__(self):
        self.tickets = []
        self.current_index = 0
        self.time_step = 0
        self.done = False

    def reset(self):
        self.current_index = 0
        self.time_step = 0
        self.done = False

        self.tickets = [
            Ticket(
                id=1,
                message="Need refund for duplicate payment",
                sentiment="angry",
                urgency="high",
                category="billing",
                sla_deadline=2,
                customer_history=5,
                resolved=False,
            ),
            Ticket(
                id=2,
                message="Order delayed by 5 days",
                sentiment="neutral",
                urgency="medium",
                category="delivery",
                sla_deadline=3,
                customer_history=2,
                resolved=False,
            ),
            Ticket(
                id=3,
                message="App crashes on login",
                sentiment="frustrated",
                urgency="high",
                category="technical",
                sla_deadline=1,
                customer_history=1,
                resolved=False,
            ),
        ]

        return Observation(
            queue=self.tickets,
            current_ticket=self.tickets[0],
            time_step=0,
            pending_count=len(self.tickets),
        )

    def step(self, action):
        ticket = self.tickets[self.current_index]

        if action.action_type == "escalate" and ticket.urgency == "high":
            reward = 0.90
        elif action.action_type == "classify":
            reward = 0.80
        elif action.action_type == "respond":
            reward = 0.70
        elif action.action_type == "close":
            reward = 0.75
            ticket.resolved = True
        else:
            reward = 0.40

        if reward <= 0:
            reward = 0.01
        elif reward >= 1:
            reward = 0.99

        self.current_index += 1
        self.time_step += 1
        self.done = self.current_index >= len(self.tickets)

        current = None if self.done else self.tickets[self.current_index]

        obs = Observation(
            queue=self.tickets,
            current_ticket=current,
            time_step=self.time_step,
            pending_count=max(0, len(self.tickets) - self.current_index),
        )

        return obs, reward, self.done, {"score": reward}

    def state(self):
        return {
            "current_index": self.current_index,
            "time_step": self.time_step,
            "done": self.done,
        }
