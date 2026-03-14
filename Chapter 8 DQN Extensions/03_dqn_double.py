import gymnasium as gym
import ptan
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from ignite.engine import Engine

from lib import dqn_model, common

NAME = "03_double"
STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100


def calc_loss_double_dqn(batch, net, tgt_net, gamma, device="cpu", double=True):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_values = net(states_v).gather(1, actions_v) 
    state_action_values = state_action_values.squeeze(-1)

    with torch.no_grad():
        next_state_values = torch.tensor(next_states).to(device)
        if double:
            # Hangi hareket en iyi (Sadece index'ini [1] alıyoruz)
            next_state_acts = net(next_state_values).max(1)[1]
            next_state_acts = next_state_acts.unsqueeze(-1)

            # Hedef Ağ - tgt_net: Seçilen hareketin değeri nedir?
            next_state_vals = tgt_net(next_state_values).gather(1, next_state_acts).squeeze(-1)
        else:
            # Standart DQN: (Hedef ağ) hem seçer hem değer biçer
            next_state_vals = tgt_net(next_state_values).max(1)[0]
        next_state_vals[done_mask] = 0.0
        exp_sa_vals = next_state_vals.detach()*gamma+rewards_v
    return nn.MSELoss()(state_action_values, exp_sa_vals)


if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--double", default=False, action="store_true", help="Enable double dqn")
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
        env, agent, gamma=params.gamma) # İçine sadece ortamı ve ajanı veriyoruz. Arka planda sonsuz döngüde oyunu oyananır ve Durum, Eylem, Ödül, Bitti mi, Yeni Durum fırlatır. while döngüsüne gerek kalmaz.
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size) # exp_source gelen verileri alır ve belirlenen kapasitede ExperienceReplayBuffer da biriktirir
    optimizer = optim.Adam(net.parameters(),
                           lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = calc_loss_double_dqn(batch, net, tgt_net.target_model,
            gamma=params.gamma, device=device, double=args.double)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration) # # Epsilon değerini bir adım daha düşür
        if engine.state.iteration % EVAL_EVERY_FRAME == 0:
            eval_states = getattr(engine.state, "eval_states", None)
            if eval_states is None:
                # Eğitim başında buffer'dan rastgele 1000 tane state alır ve dondurur, hep aynı 1000 durumu kullanacağız ve ağın o durumlara olan tepkisini görmüş olacağız
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state) for transition in eval_states]
                eval_states = np.array(eval_states)
                engine.state.eval_states = eval_states
            # Sabit 1000 durumun güncel değerini hesapla ve TensorBoard'a ("values" olarak) gönder
            engine.state.metrics["values"] = \
                common.calc_values_of_states(eval_states, net, device)
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, f"{NAME}={args.double}", extra_metrics=('values',))
    engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))