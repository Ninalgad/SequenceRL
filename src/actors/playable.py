from env import SequenceGameEnv
from typing import List
from actor import Actor
from typers import Action


def print_play_state(env: SequenceGameEnv, legal_actions: List[Action]):
    b = env.chip_board.copy().astype(int)
    b[0, 0] = b[0, 9] = b[9, 0] = b[9, 9] = -3
    for i, a in enumerate(legal_actions):
        code = -1
        if b[a.x, a.y] == env.opp:
            code = -2
        b[a.x, a.y] = code
    print("opp hand", env.hands[env.opp])
    print("plr hand", env.hands[env.actor])

    chip_map = {env.actor: '●', env.opp: '▴', 0: ' ',
                -1: '○', -2: '⊙', -3: '%'}
    state_str = "| |" + "|".join(list([str(c) for c in range(10)])) + "|\n"
    for i in range(10):
        row = f"|{i}|" + "|".join([chip_map[x] for x in b[i]]) + "|\n"
        state_str += row
    print(state_str)


def get_action_idx_from_str(s, actions):
    if len(s) != 2:
        return None

    x, y = s
    if (not x.isdecimal()) or (not y.isdecimal()):
        return None

    x, y = int(x), int(y)
    for i, a in enumerate(actions):
        if (x == a.x) and (y == a.y):
            return i
    return None


class PlayableActor(Actor):

    def reset(self):
        return

    def select_action(self, env: SequenceGameEnv, legal_actions: List[Action]):
        print_play_state(env, legal_actions)

        idx = None
        while idx is None:
            idx = get_action_idx_from_str(input('action location: '), legal_actions)
        return legal_actions[idx]
