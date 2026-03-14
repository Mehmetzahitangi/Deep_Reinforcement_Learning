import gymnasium as gym
import ptan
import argparse
import random

import torch
import torch.optim as optim

from ignite.engine import Engine

from lib import dqn_model, common

NAME = "02_n_steps"
DEFAULT_N_STEPS = 4

if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", type=int, default=DEFAULT_N_STEPS, help="Steps to do on Bellman unroll")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env) # wrap_dqn kare atlama, siyah-beyaz yapma ve 4 kareyi üst üste koyma işlemlerinin hepsini ortama uygular
    # env.seed(common.SEED)  Gymnasium'da artık ortama dışarıdan seed() verilemiyor

    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, steps_count=args.n) # İçine sadece ortamı ve ajanı veriyoruz. Arka planda sonsuz döngüde oyunu oyananır ve Durum, Eylem, Ödül, Bitti mi, Yeni Durum fırlatır. while döngüsüne gerek kalmaz.
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size) # exp_source gelen verileri alır ve belirlenen kapasitede ExperienceReplayBuffer da biriktirir
    optimizer = optim.Adam(net.parameters(),
                           lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(
            batch, net, tgt_net.target_model,
            gamma=params.gamma**args.n, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration) # # Epsilon değerini bir adım daha düşür
        if engine.state.iteration % params.target_net_sync == 0: # # Eğer zamanı geldiyse (örn: her 1000 adımda bir) Hedef Ağı güncelle
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME)
    engine.run(common.batch_generator(buffer, params.replay_initial,
                                      params.batch_size))