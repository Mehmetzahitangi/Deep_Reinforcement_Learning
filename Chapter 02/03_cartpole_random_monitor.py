"""import gymnasium as gym


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env = gym.wrappers.Monitor(env, "recording")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))
    env.close()
    env.env.close()
"""
import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")##render_mode="human")
    
    # Monitor yerine RecordVideo kullanılır (isteğe bağlıdır)
    # Video kaydetmek istemiyorsan bu satırı silebilirsin
    env = gym.wrappers.RecordVideo(env, video_folder="recording", name_prefix="cartpole-video",disable_logger=True)

    total_reward = 0.0
    total_steps = 0
    
    # 1. GÜNCELLEME: reset artık (obs, info) döndürüyor
    obs, info = env.reset()

    while True:
        action = env.action_space.sample()
        
        # 2. GÜNCELLEME: step artık 5 değer döndürüyor (terminated ve truncated)
        # 'done' yerine bu ikisinin kombinasyonu kullanılıyor
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        total_steps += 1
        
        if terminated or truncated:
            break

    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))
    
    env.close()