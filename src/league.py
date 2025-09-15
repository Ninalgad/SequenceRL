from scipy.special import softmax
import numpy as np
import json


def update_elo(winner_elo, loser_elo, k_factor=64, elo_width=400):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expected_win = expected_result(winner_elo, loser_elo, elo_width)
    change_in_elo = k_factor * (1 - expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo


def expected_result(elo_a, elo_b, elo_width):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expect_a = 1.0 / (1 + 10 ** ((elo_b - elo_a) / elo_width))
    return expect_a


class Player:
    def __init__(self, actor, rating):
        self.actor = actor
        self.rating = rating
        self.num_games = 0

    def update_rating(self, new_rating):
        self.rating = new_rating
        self.num_games += 1


class League:
    def __init__(self, init_rating=800, elo_width=400):
        self.players = {}
        self.init_rating = init_rating
        self.elo_width = elo_width

    def matchmake(self):
        pool = list(self.players.keys())
        weights = np.array([self.players[p].num_games for p in pool])
        p = softmax(0.1 * (10 - weights))
        plyr_a = np.random.choice(pool, p=p).item()
        rating_a = self.players[plyr_a].rating

        pool = [p for p in pool if (p != plyr_a)]
        ratings = np.array([self.players[p].rating for p in pool])
        weights = np.abs(ratings - rating_a) < (self.elo_width / 2)
        weights = 5 * weights.astype('float32')
        p = softmax(weights)
        plyr_b = np.random.choice(pool, p=p).item()

        if np.random.uniform() < 0.5:
            return plyr_b, plyr_a
        return plyr_a, plyr_b

    def update_ratings(self, winner_id, loser_id):
        pw, pl = self.players[winner_id], self.players[loser_id]
        elo_w, elo_l = update_elo(pw.rating, pl.rating)
        self.players[winner_id].update_rating(elo_w)
        self.players[loser_id].update_rating(elo_l)

    def register(self, actor, player_id):
        self.players[player_id] = Player(actor=actor, rating=self.init_rating)

    def best(self):
        best_rating = -np.inf
        best_player = ""
        for k, p in self.players.items():
            r = p.rating
            if r > best_rating:
                best_rating = r
                best_player = k
        return best_player

    def worst(self, min_games=10):
        worst_rating = np.inf
        worst_player = ""
        for k, p in self.players.items():
            r = p.rating
            if (r < worst_rating) and (p.num_games > min_games):
                worst_rating = r
                worst_player = k
        return worst_player

    def newest(self):
        if not len(self.players):
            return ""
        player_games = {k: v.num_games for k, v in self.players.items()}
        return min(player_games, key=player_games.get)

    def __str__(self):
        def _significance(n):
            s = ""
            if n > 10000:
                s = "***"
            elif n > 1000:
                s = "**"
            elif n > 100:
                s = "*"
            return s

        pool = list(self.players.keys())
        ratings = [int(self.players[p].rating) for p in pool]
        games = [int(self.players[p].num_games) for p in pool]
        idx = np.argsort(ratings)[::-1]
        newest = self.newest()
        s = ""
        for i in idx:
            p = pool[i]
            if p == newest:
                s += f"| {p} (latest): {ratings[i]}{_significance(games[i])} "
            else:
                s += f"| {p}: {ratings[i]}{_significance(games[i])} "

        return s

    def export_to_json(self, save_dir):
        dat = {k: {'rating': p.rating, 'num_games': p.num_games}
               for (k, p) in self.players.items()}
        with open(save_dir + 'ratings.json', 'w') as f:
            json.dump(dat, f)

    def import_json(self, save_dir):
        with open(save_dir + 'ratings.json', 'r') as f:
            dat = json.load(f)
        for pid in dat:
            r = dat[pid]['rating']
            n = dat[pid]['num_games']
            if pid in self.players:
                self.players[pid].rating = r
                self.players[pid].num_games = n
            else:
                p = Player(None, 0)
                p.rating = r
                p.num_games = n
                self.players[pid] = p
