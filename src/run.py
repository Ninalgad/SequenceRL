import pickle
import click
import os
from tqdm import tqdm
from IPython import display

from actor import Actor
from replay import ReplayBuffer
from env import SequenceGameEnv
from typers import *
from config import sequence_1v1_config
from learner import StandardLearner

from actors.dqn import DQNActor
from actors.actor_critic import A2CActor

from algorithms.dqn import DQNAlgorithm
from algorithms.actor_critic import A2CAlgorithm

from networks.cnn import ConvNetwork
from networks.mlp import MLPNetwork
from networks.vit import VitNetwork
from networks.mixer import MlpMixerNetwork


def run_n_selfplay(
        n: int, actor: Actor,
        replay_buffer: ReplayBuffer, max_turns: int = 1000):
    for _ in range(n):
        # Create a new instance of the environment.
        env = SequenceGameEnv()
        episode = []

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

        # get rewards for win/loss
        last_rewards = {c: env.reward(c) for c in Color.get_players()}

        # Send the episode to the replay.
        replay_buffer.save(episode, last_rewards)


def save(model_path, algo, replay_buffer, meta=None):
    # save algo state
    algo.model.save_weights(model_path + 'model.weights.h5')
    np.save(model_path + 'opt.npy',
            np.array(algo.optimizer.variables, dtype='object'))

    # save replays
    with open(model_path + "replay_buffer.pkl", 'wb') as f:
        config = replay_buffer.config
        replay_buffer.config = None  # can't pickle config
        pickle.dump(replay_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
    replay_buffer.config = config

    # save training info
    if meta:
        with open(model_path + "meta.pkl", 'wb') as f:
            pickle.dump(meta, f)


@click.command()
@click.option('--model-path', type=str, default="", help="Path to save the model files and trajectories.")
@click.option('--net-type', default="mlp", help="Model Architecture.", type=click.Choice(['cnn', 'mlp', 'vit', 'mix']))
@click.option('--algo-type', default="dqn", help="Learning Algorithim.", type=click.Choice(['dqn', 'a2c']))
@click.option('--resume', type=str, default="", show_default=True, help="Path to files to resume training.")
def main(model_path, net_type, algo_type, resume):
    print()
    config = sequence_1v1_config()
    if net_type == "cnn":
        net = ConvNetwork()
    elif net_type == "mlp":
        net = MLPNetwork()
    elif net_type == "vit":
        net = VitNetwork()
    else:  # net_type == "mix":
        net = MlpMixerNetwork()

    if algo_type == 'dqn':
        algo = DQNAlgorithm(net, learning_rate=config.learning_rate)
    else:
        algo = A2CAlgorithm(net, learning_rate=config.learning_rate)

    if resume:
        # load replay
        with open(resume + "replay_buffer.pkl", 'rb') as f:
            replay_buffer = pickle.load(f)
        replay_buffer.config = config

        # load algo state
        algo.build()
        algo.model.load_weights(resume + 'model.weights.h5')
        algo.optimizer.build(algo.model.trainable_variables)
        algo.optimizer.set_weights(np.load(resume + 'opt.npy', allow_pickle=True))

        # load training info
        with open(resume + "meta.pkl", 'rb') as f:
            meta = pickle.load(f)
        ep = meta['ep']

    else:
        ep = 1
        replay_buffer = ReplayBuffer(config)

    if algo_type == 'dqn':
        actor = DQNActor(algo, training=True)
    else:
        actor = A2CActor(algo, training=True)

    learner = StandardLearner(algo, config, replay_buffer)
    losses = []
    max_epochs = int(config.training_steps // config.training_steps_per_epoch)
    for ep in tqdm(range(ep, max_epochs)):
        run_n_selfplay(config.games_per_epoch, actor, replay_buffer)

        algo.train_loss.reset_state()
        for _ in range(config.training_steps_per_epoch):
            learner.learn()
        losses.append(learner.get_loss())
        display.clear_output()

        if (ep - 1) % config.export_network_every == 0:
            save(model_path, algo, replay_buffer, {'ep': ep, 'loss': np.mean(losses)})
            losses = [learner.get_loss()]
            print(f"EP: {ep}, Loss {np.mean(losses)}")

        learner.reset_loss()
        ep += 1


if __name__ == '__main__':
    main()
