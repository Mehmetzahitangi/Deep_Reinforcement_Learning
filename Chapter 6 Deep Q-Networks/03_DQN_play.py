#!/usr/bin/env python3
import gymnasium as gym
import time
import argparse
import numpy as np

import torch
import ale_py
from lib import wrappers
from lib import dqn_model

import collections

gym.register_envs(ale_py)
DEFAULT_ENV_NAME = "ALE/Pong-v5"
FPS = 25


if __name__ == "__main__": 
    """--- INFERENCE ---"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" +
                             DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", help="Directory for video")
    parser.add_argument("--no-vis", default=True, dest='vis',
                        help="Disable visualization",
                        action='store_false')
    args = parser.parse_args()

    render_mode = "rgb_array" if args.record else None
    env = wrappers.make_env(args.env, render_mode=render_mode)
    if args.record:
        env = gym.wrappers.RecordVideo(env, video_folder=args.record)
    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n)
    state = torch.load(args.model, map_location=lambda stg, _: stg)
    net.load_state_dict(state)

    state, _ = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if args.vis and not args.record:
            env.render()
        state_v = torch.tensor(np.array([state])) # O anki state'i tensor'a çevirip ağa veririz.
        q_vals = net(state_v).data.numpy()[0] # Ağ, olası 6 tuş için 6 farklı Q-değeri döndürür
        action = np.argmax(q_vals) # Değerler içindeki en yüksek olanı seçeriz
        c[action] += 1 # Oyun bittiğinde ajanın hangi tuşlara ne kadar bastığını gösterir. Eğitimsiz bir ajan genelde takılı kalıp hep aynı tuşa basar. Eğitimli bir ajan ise tuşları duruma göre dengeli kullanır.
        state, reward, terminated, truncated , _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if done:
            break
        if args.vis:
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta) # Ajan oyunu saniyede binlerce kare hızında oynayabilir. Eğer oyunu ekranda izlemek (--vis) istersek, çok hızlı akıp bitmemesi için saniyede 25 kareye (FPS) sabitleriz.
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        env.close()