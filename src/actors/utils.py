import numpy as np


def collate_states(states):
    x = states.pop(0)
    x = {k: [v] for (k, v) in x.items()}
    for s in states:
        for k, v in s.items():
            x[k].append(v)

    return x
