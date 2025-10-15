# run_trained_dqn.py
import torch
import numpy as np
from rocket_env import SimpleRocketEnv
from dqn_pytorch import QNetwork

# Siapkan device dan environment dengan render tampilan
device = "cuda" if torch.cuda.is_available() else "cpu"
env = SimpleRocketEnv(render_mode="human")

# Ambil ukuran input & aksi
state, _ = env.reset()
obs_dim = state.shape[0]
n_actions = env.action_space.n

# Load model hasil training
model = QNetwork(obs_dim, n_actions)
model.load_state_dict(torch.load("dqn_pytorch_final.pt", map_location=device))
model.to(device)
model.eval()

# Jalankan episode
done = False
total_reward = 0
while not done:
    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = model(s).argmax(1).item()
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    env.render()

env.close()
print(f"Total reward episode: {total_reward:.2f}")
