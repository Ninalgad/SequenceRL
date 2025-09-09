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
TWO_EYED_JACKS = ['jc', 'jd', 'jc', 'jd']
ONE_EYED_JACKS = ['js', 'jh', 'js', 'jh']
CORNERS = [(0, 0), (0, 9), (9, 0), (9, 9)]


def get_deck():
    deck = []
    for i in ['a'] + list(range(2, 11)):
        for s in ['s', 'h', 'd', 'c']:
            c = str(i) + s
            assert (c == BOARD).sum() == 2, c
            deck += [c, c]
    deck += TWO_EYED_JACKS + ONE_EYED_JACKS
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
    grid = np.array(grid, dtype=int)
    n = grid.shape[0]
    counts = {1: 0, 2: 0}
    sequences = {1: [], 2: []}  # store coordinates of completed sequences

    def add_sequence(val, coords):
        length = len(coords)
        if length < 5:
            return
        if length >= 10:
            counts[val] += 2
        else:  # length 5–9
            counts[val] += 1
        sequences[val].append(coords)

    def process_line(line, coords):
        if len(line) < 5:
            return
        diffs = np.diff(line)
        run_ends = np.where(diffs != 0)[0] + 1
        run_starts = np.r_[0, run_ends]
        run_ends = np.r_[run_ends, len(line)]
        run_vals = line[run_starts]
        run_lengths = run_ends - run_starts

        for val, start, end in zip(run_vals, run_starts, run_ends):
            if val == 0 or (end - start) < 5:
                continue
            coords_run = coords[start:end]
            add_sequence(val, coords_run)

    def process_edge_line(line, coords):
        assert len(line) == 10
        line = line.copy()
        # copy neighbors into corners
        line[0] = line[1]
        line[-1] = line[-2]
        process_line(line, coords)

    # Rows and columns
    for i in range(1, n-1):
        process_line(grid[i, :], [(i, j) for j in range(n)])
        process_line(grid[:, i], [(j, i) for j in range(n)])

    process_edge_line(grid[0, :], [(0, j) for j in range(n)])
    process_edge_line(grid[-1, :], [(n-1, j) for j in range(n)])
    process_edge_line(grid[:, 0], [(i, 0) for i in range(n)])
    process_edge_line(grid[:, -1], [(i, n-1) for i in range(n)])

    process_edge_line(np.diag(grid), [(i, i) for i in range(n)])
    process_edge_line(np.diag(np.fliplr(grid)), [(i, n-1-i) for i in range(n)])

    # Diagonals (↘) and anti-diagonals (↙) off main
    for offset in range(-n + 1, n):
        if offset != 0:
            diag = np.diag(grid, k=offset)
            coords = [(i, i+offset) for i in range(len(diag))] if offset >= 0 \
                     else [(i-offset, i) for i in range(len(diag))]
            process_line(diag, coords)

            diag = np.diag(np.fliplr(grid), k=offset)
            coords = [(i, n-1-i-offset) for i in range(len(diag))] if offset >= 0 \
                     else [(i-offset, n-1-i) for i in range(len(diag))]
            process_line(diag, coords)

    if return_sequences:
        return counts, sequences
    return counts


def get_opp(color):
    if color == Color.RED:
        return Color.BLUE
    return Color.RED


def board_repr(chip_board, actor, opp):
    a = np.expand_dims(chip_board == actor, -1)
    o = np.expand_dims(chip_board == opp, -1)
    return np.concatenate([a, o], axis=-1)
