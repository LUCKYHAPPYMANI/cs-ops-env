print("LOADED REAL graders.py")


def strict(score):
    try:
        score = float(score)
    except Exception:
        score = 0.5

    if score <= 0:
        return 0.01
    if score >= 1:
        return 0.99

    return score


def grade_easy(pred, true):
    if not true:
        return 0.51

    correct = sum(1 for p, t in zip(pred, true) if p == t)
    raw = correct / max(len(true), 1)
    return strict(raw)


def grade_medium(actions):
    if not actions:
        return 0.52

    handled = sum(
        1 for a in actions
        if a.get("handled_before_deadline", False)
    )
    raw = handled / max(len(actions), 1)
    return strict(raw)


def grade_hard(trajectory):
    if not trajectory:
        return 0.53

    rewards = []
    for step in trajectory:
        try:
            rewards.append(float(step.get("reward", 0.5)))
        except Exception:
            rewards.append(0.5)

    raw = sum(rewards) / max(len(rewards), 1)
    return strict(raw)