import numpy as np
from typers import *


BOARD = [
    ['x', '10s', 'qs', 'ks', 'as', '2d', '3d', '4d', '5d', 'x'],
    ['9s', '10h', '9h', '8h', '7h', '6h', '5h', '4h', '3h', '6d'],
    ['8s', 'qs', '7d', '8d', '9d', '10d', 'qd', 'kd', '2h', '7d'],
    ['7s', 'kh', '6d', '2c', 'ah', 'kh', 'qh', 'ad', '2s', '8d'],
    ['6s', 'ah', '5d', '3c', '4h', '3h', '10h', 'ac', '3s', '9d'],
    ['5s', '2c', '4d', '4c', '5h', '2h', '9h', 'ks', '4s', '10d'],
    ['4s', '3c', '3d', '5c', '6h', '7h', '8h', 'qc', '5s', 'qd'],
    ['3s', '4c', '2d', '6c', '7c', '8c', '9c', '10c', '6s', 'kd'],
    ['2s', '5c', 'as', 'ks', 'qs', '10s', '9s', '8s', '7s', 'ad'],
    ['x', '6c', '7c', '8c', '9c', '10c', 'qc', 'kc', 'ac', 'x'],
]
BOARD = np.array(BOARD)
TWO_EYED_JACKS = ['jc', 'jd']
ONE_EYED_JACKS = ['js', 'jh']


def get_deck():
    deck = []
    for i in ['a'] + list(range(2, 11)):
        for s in ['s', 'h', 'd', 'c']:
            c = str(i) + s
            assert (c == BOARD).sum() == 2, c
            deck += [c, c]
    deck += 2 * TWO_EYED_JACKS + 2 * ONE_EYED_JACKS
    return deck


def card_set_repr(cards):
    cards = set(cards)
    x = [c in cards for c in get_deck()]
    return x


def legal_actions_repr(actions):
    board = np.zeros((10, 10, 1), dtype='uint8')
    if actions is not None:
        for a in actions:
            board[a.x][a.y] = 1
    return board


def is_valid_position(x, y, n=10):
    return 0 <= x < n and 0 <= y < n


def unique_sequences(grid, return_sequences=False):
    grid = grid.copy()
    directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
    empty = 0

    def get_sequence(x, y, dx, dy, colour):
        sequence = []
        n = 0
        while is_valid_position(x, y) and (colour == grid[x][y]):
            sequence.append((x, y))
            x += dx
            y += dy
            n += 1
        if n < 5:
            return []
        return sequence

    sequence_counts = {1: 0, 2: 0}
    used = []
    completed = []
    for x in range(10):
        for y in range(10):
            c = grid[x][y]
            if (c != empty) and ((x, y) not in used):
                grid[0][9] = grid[0][0] = grid[9][0] = grid[9][9] = grid[0][9] = c
                for dx, dy in directions:
                    sequence = get_sequence(x, y, dx, dy, c)
                    used += sequence
                    n = len(sequence)
                    if n >= 7:
                        sequence_counts[c] += 2
                        completed += sequence
                    elif n >= 4:
                        sequence_counts[c] += 1
                        completed += sequence

    if return_sequences:
        return sequence_counts, completed
    return sequence_counts


def get_opp(color):
    if color == Color.RED:
        return Color.BLUE
    return Color.RED


def board_repr(chip_board, actor, opp):
    a = np.expand_dims(chip_board == actor, -1)
    o = np.expand_dims(chip_board == opp, -1)
    return np.concatenate([a, o], axis=-1)
