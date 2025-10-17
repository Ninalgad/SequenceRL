from scipy.special import softmax
import numpy as np
import json
import heapq


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


def _scale_rating(r, mean, std):
    return 1.2 * (r - mean) / (std + 1e-10)


class League:
    def __init__(self, init_rating=800, elo_width=400):
        self.players = {}
        self.init_rating = init_rating
        self.elo_width = elo_width
        self.newest = None

    def matchmake(self):
        # max_rating = self.players[self.best()].rating
        # min_rating = self.players[self.worst()].rating
        ratings = [p.rating for p in self.players.values()]
        m, s = np.mean(ratings), np.std(ratings)

        # [0-5]
        player2weight = {k: _scale_rating(v.rating, m, s)
                         for k, v in self.players.items()}

        if self.newest in self.players:
            # 10 placement games for the latest player (against the best players)
            if self.players[self.newest].num_games < 10:
                player2weight[self.newest] = 100
            else:
                player2weight[self.newest] += 1

        # add noise
        player2weight = {k: v + np.random.gumbel() for k, v in player2weight.items()}

        # select top 2 by weight
        plyr_a, plyr_b = heapq.nlargest(2, player2weight, key=player2weight.get)

        # randomly set order
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
        self.newest = player_id

    def best(self):
        if len(self.players):
            return max(self.players, key=lambda x: self.players[x].rating)
        return None

    def worst(self):
        def _worst_not_newest(x):
            r = self.players[x].rating
            if x == self.newest:
                r = np.inf
            return r

        if len(self.players):
            return min(self.players, key=_worst_not_newest)
        return None

    def newest(self):
        if not len(self.players):
            return ""
        player_games = {k: v.num_games for k, v in self.players.items()}
        return min(player_games, key=player_games.get)

    def __str__(self):
        def _significance(n):
            sig = ""
            if n > 10000:
                sig = "***"
            elif n > 1000:
                sig = "**"
            elif n > 100:
                sig = "*"
            return sig

        pool = list(self.players.keys())
        ratings = [int(self.players[p].rating) for p in pool]
        games = [int(self.players[p].num_games) for p in pool]
        idx = np.argsort(ratings)[::-1]
        newest = self.newest
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
