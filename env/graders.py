def grade_easy(pred, true):
    return round(sum(p == t for p, t in zip(pred, true)) / len(true), 2)

def grade_medium(actions):
    return round(sum(a["handled_before_deadline"] for a in actions) / len(actions), 2)

def grade_hard(trajectory):
    total = sum(step["reward"] for step in trajectory)
    return round(min(total / len(trajectory), 1.0), 2)