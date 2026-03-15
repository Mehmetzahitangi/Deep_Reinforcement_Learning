import gymnasium as gym
import torch
import ptan
import numpy as np

from lib import dqn_extra


# render_mode="rgb_array" videoya çekebilmek için zorunlu
env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
env = ptan.common.wrappers.wrap_dqn(env)

# Videolar "video_klasoru" içine kaydedilecek.
env = gym.wrappers.RecordVideo(env, video_folder="video_klasoru", name_prefix="rainbow_sampiyon")


# Eğitirken kullanılan mimarinin aynısı olmak zorunda
net = dqn_extra.RainbowDQN(env.observation_space.shape, env.action_space.n)

# Kaydettiğimiz modeli  yüklüyoruz
net.load_state_dict(torch.load("rainbow_pong_model.dat", map_location="cpu", weights_only=True))
net.eval() # Ağı "test/oyun" moduna alıyoruz (eğitimi durdur)

# OYUN DÖNGÜSÜ
state, info = env.reset()
total_reward = 0.0

print("Oyun başlıyor! Video kaydediliyor...")

while True:
    # Görüntüyü PyTorch tensorüne çevir ve "batch" boyutu ekle
    state_v = torch.tensor(np.array([state], copy=False)).to("cpu")
    
    # qvals fonksiyonu RainbowDQN içinde ortalama Q değerlerini döner
    q_vals = net.qvals(state_v)[0]
    
    # Epsilon (zar atma) yok. En yüksek Q değerine sahip hareketi (argmax) seç
    action = torch.argmax(q_vals).item()
    
    # Hareketi oyunda yap
    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        break

print(f"Oyun bitti! Ajanın Toplam Skoru: {total_reward}")
env.close()
print("Video başarıyla 'video_klasoru' klasörüne kaydedildi!")