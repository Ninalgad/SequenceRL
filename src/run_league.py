import click
import os
import json
from tqdm import tqdm

from actor import Actor
from replay import ReplayBuffer
from env import SequenceGameEnv
from typers import *
from config import sequence_1v1_league_config
from learner import StandardLearner

from actors.dqn import DQNActor
from algorithms.dqn import DQNAlgorithm
from networks.mlp import MLPNetwork

from league import League


def run_n_selfplay(
        n: int, actor: Actor,
        replay_buffer: ReplayBuffer, max_turns: int = 1000):
    for _ in range(n):
        # Create a new instance of the environment.
        env = SequenceGameEnv()
        episode = []

        try:
            i = 0
            while (not env.is_terminal()) and (i < max_turns):
                legal_actions = env.legal_actions()

                action = actor.select_action(env, legal_actions)

                state = State(
                    observation=env.observation(),
                    reward=env.reward(env.to_play()),
                    player=env.to_play(),
                    action=action
                )
                episode.append(state)
                env.apply(action)
                i += 1
        except (IndexError, ValueError):
            continue

        # get rewards for win/loss
        last_rewards = {c: env.reward(c) for c in Color.get_players()}

        # Send the episode to the replay.
        replay_buffer.save(episode, last_rewards)


def run_n_leagueplay(
        n: int, league: League,
        replay_buffer: ReplayBuffer, max_turns: int = 1000):
    n = max(n, 2)
    for j in range(n):
        a_id, b_id = league.matchmake()
        actors = [league.players[a_id].actor, league.players[b_id].actor]

        # Create a new instance of the environment.
        env = SequenceGameEnv()
        episode = []

        try:
            i = 0
            while (not env.is_terminal()) and (i < max_turns):
                legal_actions = env.legal_actions()
                actor = actors[i % 2]
                action = actor.select_action(env, legal_actions)

                state = State(
                    observation=env.observation(),
                    reward=env.reward(env.to_play()),
                    player=env.to_play(),
                    action=action
                )
                episode.append(state)
                env.apply(action)
                i += 1
        except (IndexError, ValueError):
            continue

        # get rewards for win/loss
        last_rewards = {c: env.reward(c) for c in Color.get_players()}

        # Send the episode to the replay.
        replay_buffer.save(episode, last_rewards)

        # update league ratings based on results
        if env.winner() == Color.BLUE:
            league.update_ratings(winner_id=a_id, loser_id=b_id)
        else:
            league.update_ratings(winner_id=b_id, loser_id=a_id)


def recreate_league_from_dir(league_dir, config):
    L = League()
    L.import_json(league_dir)
    for lid in L.players:
        new_actor_weights = league_dir + f'{lid}.weights.h5'
        new_actor = create_dqn_bot(new_actor_weights, config)
        L.players[lid].actor = new_actor
    return L


def save(model_path, algo, meta):
    # save algo state
    algo.model.save_weights(model_path + 'model.weights.h5')
    np.save(model_path + 'opt.npy',
            np.array(algo.optimizer.variables, dtype='object'))

    # save training info
    with open(model_path + 'train.json', 'w') as f:
        json.dump(meta, f)


def create_dqn_bot(weight_file, config):
    net = MLPNetwork()
    algo = DQNAlgorithm(net, learning_rate=config.learning_rate)
    algo.build()
    del algo.optimizer
    algo.model.load_weights(weight_file)
    actor = DQNActor(algo, training=False)
    return actor


@click.command()
@click.option('--model-path', type=str, default="", help="Path to save the model files and trajectories.")
@click.option('--league-dir', type=str, default="", help="Path to save the league models.")
@click.option('--init-weights', type=str, default="", help="Path to model prior weights.")
@click.option('--resume', type=str, default="", show_default=True, help="Path to files to resume training.")
def main(model_path, league_dir, init_weights, resume):
    print()
    config = sequence_1v1_league_config()
    net = MLPNetwork()
    algo = DQNAlgorithm(net, learning_rate=config.learning_rate)
    algo.build()

    if resume:

        # load algo state
        print('loading algo')
        algo.model.load_weights(resume + 'model.weights.h5')
        algo.optimizer.build(algo.model.trainable_variables)
        algo.optimizer.set_weights(np.load(resume + 'opt.npy', allow_pickle=True))

        # load training info
        with open(resume + 'train.json', 'r') as f:
            meta = json.load(f)
        ep = meta['ep']

        # load league
        print('loading league')
        if os.path.isdir(league_dir):
            L = recreate_league_from_dir(league_dir, config)
            assert len(L.players) > 0
            print(f"Loaded league of size {len(L.players)} from {league_dir}")

        else:
            L = League()
            print("Creating new league.")

    else:
        ep = 1
        L = League()

        if init_weights:
            algo.model.load_weights(init_weights)

    actor = DQNActor(algo, training=True)
    replay_buffer = ReplayBuffer(config)
    learner = StandardLearner(algo, config, replay_buffer)
    losses = []
    max_epochs = int(config.training_steps // config.training_steps_per_epoch)

    pbar = tqdm(range(ep, max_epochs))

    for ep in pbar:
        run_n_selfplay(config.self_play_games_per_epoch, actor, replay_buffer)

        if len(L.players) > 3:
            run_n_leagueplay(config.run_n_leagueplay_games_per_epoch, L, replay_buffer)

        algo.train_loss.reset_state()
        for _ in range(config.training_steps_per_epoch):
            learner.learn()
        losses.append(learner.get_loss())
        replay_buffer.reset()

        pbar.set_description(f"League Stats: {str(L)}")

        if (ep - 1) % config.export_network_every == 0:
            save(model_path, algo, {'ep': ep, 'loss': np.mean(losses).item()})
            losses = [learner.get_loss()]

            # add new league entry
            league_id = f"ep{ep}"
            new_actor_weights = league_dir + f'{league_id}.weights.h5'
            algo.model.save_weights(new_actor_weights)
            new_actor = create_dqn_bot(new_actor_weights, config)
            L.register(new_actor, league_id)

            # remove worst performing
            if len(L.players) > config.max_league_size:
                worst_lid = L.worst()
                if worst_lid:
                    os.remove(league_dir + f'{worst_lid}.weights.h5')
                    del L.players[worst_lid]
                    print(f"Removed {worst_lid} from the league.")
            # save league data
            L.export_to_json(league_dir)
            print(f"EP: {ep}, Loss {np.mean(losses)}")

        learner.reset_loss()
        ep += 1


if __name__ == '__main__':
    main()
