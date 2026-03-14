import numpy as np
import torch
import torch.nn as nn
import warnings
from datetime import timedelta, datetime
from types import SimpleNamespace
from typing import Iterable, Tuple, List

import ptan
import ptan.ignite as ptan_ignite
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger


SEED = 123

HYPERPARAMS = {
    'pong': SimpleNamespace(**{
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    }),
    'breakout-small': SimpleNamespace(**{
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout-small',
        'replay_size':      3*10 ** 5,
        'replay_initial':   20000,
        'target_net_sync':  1000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       64
    }),
    'breakout': SimpleNamespace(**{
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout',
        'replay_size':      10 ** 6,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32
    }),
    'invaders': SimpleNamespace(**{
        'env_name': "SpaceInvadersNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'breakout',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    }),
}


def unpack_batch(batch:List[ptan.experience.ExperienceFirstLast]):
    """
    PTAN ajanın oynadığı her adımı özel bir nesne (ExperienceFirstLast) olarak tutar.
    Ama bizim sinir ağımız nesnelerden değil, PyTorch Tensor ve NumPy ile işlem yapar. Bu metot, PTAN nesnelerini ayrı ayrı NumPy listelerine dönüştürür.
    """
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        state = np.array(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = state # the result will be masked anyway
        else:
            lstate = np.array(exp.last_state)
        last_states.append(lstate)
    return np.array(states), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(last_states)


def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    """
    Ajanın tahmin ettiği Q değerleri ile Target Network'ün öngördüğü gelecekteki Q değerleri karşılaştırır.
    """
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device) # Gerçek ödül
    done_mask = torch.BoolTensor(dones).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_values = net(states_v).gather(1, actions_v) # Hesaplanan tüm değerler arasından sadece bizim yaptığımız aksiyonun çekip alır, bu mevcut tahminimizdir
    state_action_values = state_action_values.squeeze(-1)

    with torch.no_grad(): # DQN'de next state'e asıl ağ ile değil Target Network ile bakarız, Hedef ağın tahmin ettiği en yüksek değeri alırız. Eğer oyun o adımda bittiyse (done_mask), gelecekte ödül yoktur, değeri sıfırlarız.
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0


    bellman_vals = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, bellman_vals) # Tahminimiz ile Olması gereken arasındaki fark

class EpsilonTracker:
    """
    Ajanın keşfetme oranını(rastgelelik) frame sayısına göre lineer olarak düşürür. 
    epsilon değerini güncelleyerek ajanın yavaş yavaş kendine güvenmesini sağlar.
    """
    def __init__(self, 
                 selector: ptan.actions.EpsilonGreedyActionSelector,
                 params: SimpleNamespace):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - frame_idx / self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final,eps)

def batch_generator(buffer:ptan.experience.ExperienceReplayBuffer,
                    initial: int, batch_size: int):
    buffer.populate(initial) # Experience'i doldurur önce 
    while True: # Her çağırmada yeni bir frame ekler ve eğitim için bir batch fırlatır
        buffer.populate(1)
        yield buffer.sample(batch_size)

def setup_ignite(engine: Engine, params: SimpleNamespace,
                 exp_source, run_name: str,
                 extra_metrics: Iterable[str] = ()):
    """
    PyTorch Ignite, Olay Güdümlü (Event-Driven) çalışır.
    """
    warnings.simplefilter("ignore", category=UserWarning)
    handler = ptan_ignite.EndOfEpisodeHandler(
        exp_source, bound_avg_reward=params.stop_reward)
    handler.attach(engine)
    ptan_ignite.EpisodeFPSHandler().attach(engine)

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED) #Bir episode bittiğinde tetiklenir ve ekrana Episode X: reward=Y... şeklinde log basar
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        print("Episode %d: reward=%.0f, steps=%s, "
                "speed=%.1f f/s, elapsed=%s" % (
                    trainer.state.episode, trainer.state.episode_reward,
                    trainer.state.episode_steps,
                    trainer.state.metrics.get('avg_fps', 0),
                    timedelta(seconds=int(passed))))
        
    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED) # Hedef puana ulaşıldığında tetiklenir, eğitimi otomatik durdurur (trainer.should_terminate = True)
    def game_solved(trainer: Engine):
        passed = trainer.state.metrics['time_passed']
        print("Game solved in %s, after %d episodes "
                "and %d iterations!" % (
            timedelta(seconds=int(passed)),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    now = datetime.now().isoformat(timespec='minutes').replace(':', '')
    logdir = f"runs/{now}-{params.run_name}-{run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir) # tb_logger ile her 100 adımda bir loss değerini ve hızı (avg_fps) otomatik olarak TensorBoard grafiklerine çizer.
    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, "avg_loss")

    metrics = ['reward', 'steps', 'avg_reward']
    handler = tb_logger.OutputHandler(
        tag="episodes", metric_names=metrics)
    event = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # write to tensorboard every 100 iterations
    ptan_ignite.PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    handler = tb_logger.OutputHandler(
        tag="train", metric_names=metrics,
        output_transform=lambda a: a)
    event = ptan_ignite.PeriodEvents.ITERS_100_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)


def calc_values_of_states(states, net, device="cpu"):
    mean_vals = [] 
    
    # 1000 adet durumu (state, oyun ekranını) tek seferde GPU'ya atmak RAM'i şişirebilir, 
    # bu yüzden 64'lük paketlere bölüyoruz (array_split)
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)

        # O durum için öngörülen en yüksek Q-değerini (best_action_values_v) al
        best_action_values_v = action_values_v.max(1)[0]

        # Ortalama Q-değerini listeye ekle (ortalama olarak kaç puan kazanacağız)
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)