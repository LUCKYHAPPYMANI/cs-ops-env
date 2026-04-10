from fastapi import FastAPI
import threading
import time
import uvicorn

from env.environment import CustomerSupportEnv
from env.models import Action

app = FastAPI()
env = CustomerSupportEnv()


def run_tasks():
    time.sleep(2)

    for task in ["easy", "medium", "hard"]:
        obs = env.reset()

        print(f"[START] task={task}", flush=True)

        for step in range(3):
            ticket = obs.current_ticket

            action = Action(
                action_type="respond",
                ticket_id=ticket.id,
                content="Auto response"
            )

            obs, reward, done, _ = env.step(action)

            print(f"[STEP] step={step+1} reward={reward:.2f}", flush=True)

            if done:
                break

        print(f"[END] task={task}", flush=True)


# ✅ START BACKGROUND TASKS
@app.on_event("startup")
def start():
    threading.Thread(target=run_tasks).start()


# ✅ REQUIRED ROOT ENDPOINT
@app.get("/")
def home():
    return {"status": "running"}


# 🔥 CRITICAL FIX → RESET ENDPOINT
@app.post("/reset")
def reset():
    env.reset()
    return {"status": "reset successful"}


# OPTIONAL (SAFE)
@app.post("/step")
def step():
    return {"status": "step placeholder"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()