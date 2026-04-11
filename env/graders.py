def safe_score(score):
    if score <= 0:
        return 0.01
    elif score >= 1:
        return 0.99
    return round(score, 2)


def grade_easy(pred, true):
    if not true:
        return 0.5
    raw = sum(p == t for p, t in zip(pred, true)) / len(true)
    return safe_score(raw)


def grade_medium(actions):
    if not actions:
        return 0.5
    raw = sum(a.get("handled_before_deadline", 0) for a in actions) / len(actions)
    return safe_score(raw)


def grade_hard(trajectory):
    if not trajectory:
        return 0.5
    raw = sum(step.get("reward", 0) for step in trajectory) / len(trajectory)
    return safe_score(raw)