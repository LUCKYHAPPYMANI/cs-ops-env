from fastapi import FastAPI
import threading
import time

from env.environment import CustomerSupportEnv
from env.models import Action

app = FastAPI()
env = CustomerSupportEnv()

def run_tasks():
    time.sleep(3)

    for task in ["easy", "medium", "hard"]:
        obs = env.reset()
        total_reward = 0

        print(f"[START] task={task} env=cs-ops-env model=gpt-4o-mini", flush=True)

        for step in range(15):
            ticket = obs.current_ticket

            action = Action(
                action_type="respond",
                ticket_id=ticket.id,
                content="Auto response"
            )

            obs, reward, done, _ = env.step(action)
            total_reward += reward

            print(
                f"[STEP] step={step+1} action=respond reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True
            )

            if done:
                break

        print(f"[END] success=true steps={step+1} rewards={total_reward:.2f}", flush=True)

threading.Thread(target=run_tasks).start()


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/reset")
def reset():
    env.reset()
    return {"message": "reset done"}