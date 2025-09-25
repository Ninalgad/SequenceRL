from actor import Actor
from algorithm import Algorithm
from .search import *
from .typers import *


class StochasticMuZeroActor(Actor):
    def __init__(self, config, algo: Algorithm, training: bool = False, verbose: bool = False):
        super(StochasticMuZeroActor, self).__init__()
        self.network = algo.model
        self.algo = algo
        self.training = training
        self.num_actions = 0
        self.verbose = verbose
        self.config = config
        self.root = None

    def reset(self):
        self.num_actions = 0
        self.root = None

    def select_action(self, env, actions):
        self.num_actions += 1
        if len(actions) == 1:
            return actions[0]
        obs = env.observation()
        # New min max stats for the search tree.
        min_max_stats = MinMaxStats(self.config.known_bounds)

        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)

        # Provide the history of observations to the representation network to
        # get the initial latent state.
        latent_state = self.network.representation(**obs)
        # Compute the predictions.
        outputs = self.network.predictions(latent_state)

        # Keep only the legal actions.
        outputs = self._mask_illegal_actions(actions, outputs)

        # Expand the root node.
        expand_node(root, latent_state, outputs, env.to_play(), is_chance=False)

        # Backpropagate the value.
        backpropagate([root], outputs.value, env.to_play(),
                      self.config.discount, min_max_stats)
        # We add exploration noise to the root node.
        add_exploration_noise(self.config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(self.config, root, ActionOutcomeHistory(env.to_play()),
                 self.network, min_max_stats, verbose=self.verbose)

        # Keep track of the root to return the stats.
        self.root = root

        # Return an action.
        return self._select_action(root, actions)

    def _select_action(self, root, actions):
        """Selects an action given the root node."""
        # Get the visit count distribution.
        actions_loc, visit_counts = zip(*[
            (action, node.visit_count)
            for action, node in root.children.items()
        ])

        # Temperature
        temperature = self.config.visit_softmax_temperature_fn(self.algo.training_step)

        # Compute the search policy.
        search_policy = [v ** (1. / temperature) for v in visit_counts]
        norm = sum(search_policy)
        search_policy = [v / norm for v in search_policy]
        loc = actions_loc[np.random.choice(len(actions_loc), p=search_policy)]
        for act in actions:
            if loc == (act.x, act.y):
                return act
        return None

    def _mask_illegal_actions(self, legal_actions, outputs):
        """Masks any actions which are illegal at the root."""
        # We mask out and keep only the legal actions.
        masked_policy = {}
        network_policy = outputs.probabilities

        norm = 0
        for action in legal_actions:
            key = (action.x.item(), action.y.item())
            if key not in masked_policy:
                masked_policy[key] = network_policy[key]
            else:
                masked_policy[key] = 0.0
            norm += masked_policy[key]

        # Renormalize the masked policy.
        masked_policy = {a: v / norm for a, v in masked_policy.items()}
        return NetworkOutput(value=outputs.value, probabilities=masked_policy)
