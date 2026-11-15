"""Immediate value-minus-cost baseline from the bundle notebook."""


def choose_action(state):
    seat = state["seat"]

    def value(action):
        if action is None:
            return min(3, 9 - state["energy"][seat]) * 1.5
        capture = state["owners"][action] == 1 - seat
        gain = state["weights"][action] * (2 if capture else 1)
        cost = state["costs"][action] + (2 if capture else 0)
        return gain - 1.5 * cost
    return max(state["legal_actions"], key=value)
