from game_utils import *
from typers import Action


class SequenceGameEnv:
    """Implements the rules of the environment."""

    def __init__(self):

        deck = get_deck()
        np.random.shuffle(deck)
        self.deck = deck
        self.discarded = []
        self.chip_board = np.zeros_like(BOARD, dtype='uint8')
        self.card_board = BOARD.copy()
        self.hands = {Color.BLUE: self.deck[:7], Color.RED: self.deck[7:14]}

        self.deck = self.deck[14:]
        self.actor = Color.BLUE
        self.opp = Color.RED

    def _swap_players(self):
        self.actor = get_opp(self.actor)
        self.opp = get_opp(self.opp)

    def apply(self, action: Action):
        """Applies an action or a chance outcome to the environment."""
        x, y, played_card = action.x, action.y, action.card
        colour = 0 if played_card in ONE_EYED_JACKS else self.actor
        self.chip_board[x][y] = colour

        # remove card from hand
        cards = self.hands[self.actor]
        del self.hands[self.actor][cards.index(played_card)]

        # discard
        self.discarded.append(played_card)

        # draw
        if self.deck:
            self.hands[self.actor].append(self.deck.pop(0))

        # switch players
        self._swap_players()

        self._replenish_deck()

    def observation(self, reverse: bool = False):
        """Returns the observation of the environment to feed to the network."""
        actor, opp = self.actor, self.opp
        if reverse:
            actor = get_opp(self.actor)
            opp = get_opp(self.opp)

        board_obs = board_repr(self.chip_board, actor, opp).astype('uint8')
        hand_obs = card_set_repr(self.hands[actor])

        scores = unique_sequences(self.chip_board)
        sorted_keys = sorted(scores.keys())
        score_obs = np.array([scores[k] for k in sorted_keys], 'uint8')

        vec_obs = np.concatenate([hand_obs, score_obs], dtype='uint8')

        return {'board': board_obs, 'vec': vec_obs}

    def winner(self):
        scores = unique_sequences(self.chip_board)
        leading = max(scores, key=scores.get)
        if scores[leading] >= 2:
            return leading
        return None

    def is_terminal(self) -> bool:
        """Returns true if the environment is in a terminal state."""
        # draw = self.legal_actions() == []
        decisive = max(unique_sequences(self.chip_board).values()) >= 2
        return decisive  # or draw

    def _replenish_deck(self):
        if not self.deck:
            self.deck = self.discarded
            self.discarded = []
            np.random.shuffle(self.deck)

    def _legal_actions(self):
        actions = []
        blocked_pos = []
        for card in self.hands[self.actor]:
            if card in ONE_EYED_JACKS:  # remove
                placement = self.opp
                blocked_pos = unique_sequences(self.chip_board, True)[1]
            elif card in TWO_EYED_JACKS:  # place anywhere
                placement = 0
            else:
                placement = card

            blocked_pos = set(blocked_pos)
            pos = np.where(self.card_board == placement)

            for x, y in zip(*pos):
                if (x, y) not in blocked_pos:
                    act = Action(x, y, card)
                    actions.append(act)

        return actions

    def legal_actions(self):
        """Returns the legal actions for the current state."""
        n_cards_left = len(self.discarded) + len(self.deck)
        actions = self._legal_actions()
        i = 0
        while (actions == []) and (i < n_cards_left):
            # remove card from hand at random
            cards = self.hands[self.actor]
            idx = np.random.choice(len(cards))
            removed_card = self.hands[self.actor][idx]
            del self.hands[self.actor][idx]

            # dicard
            self.discarded.append(removed_card)

            # refill deck
            self._replenish_deck()

            # draw
            if self.deck:
                self.hands[self.actor].append(self.deck.pop(0))

            actions = self._legal_actions()
            i += 1

        return actions

    def reward(self, color) -> float:
        """Returns the last reward for the player."""
        # only win based reward
        scores = unique_sequences(self.chip_board)
        if max(scores.values()) >= 2:
            winner = max(scores, key=scores.get)
            return 2 * (int(winner == color) - .5)  # [-1, 1]
        return 0

    def to_play(self):
        """Returns the current player to play."""
        return self.actor
