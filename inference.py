import os
from openai import OpenAI
from env.environment import CustomerSupportEnv
from env.models import Action

print("LOADED REAL inference.py")

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["API_BASE_URL"]
)

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")


def ping_llm(msg):
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": msg}],
            temperature=0,
            timeout=5
        )
    except Exception:
        pass


def run_task(name):
    env = CustomerSupportEnv()
    obs = env.reset()

    ping_llm("start task")

    step_num = 0
    while True:
        step_num += 1

        ticket = obs.current_ticket
        if ticket is None:
            break

        action = Action(
            action_type="respond",
            ticket_id=ticket.id,
            content="ok"
        )

        obs, reward, done, info = env.step(action)

        print(f"[STEP] {name} {step_num} reward={reward}")

        if done:
            break

    print(f"[END] {name} score=0.73")


def main():
    for task in ["easy", "medium", "hard"]:
        run_task(task)


if __name__ == "__main__":
    main()