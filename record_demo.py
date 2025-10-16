import torch
import numpy as np
import imageio
import pygame
from rocket_env import SimpleRocketEnv
from dqn_pytorch import QNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"
env = SimpleRocketEnv(render_mode="human")

state, _ = env.reset()
obs_dim = state.shape[0]
n_actions = env.action_space.n

model = QNetwork(obs_dim, n_actions)
model.load_state_dict(torch.load("./DQN_PytorchFinal/dqn_pytorch_ep2000.pt", map_location=device))
model.to(device)
model.eval()

frames = []
done = False
total_reward = 0

while not done:
    env.render()
    frame = pygame.surfarray.array3d(env.screen)
    frame = np.transpose(frame, (1, 0, 2))  # ubah dari (W,H,3) ke (H,W,3)
    frames.append(frame)

    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = model(s).argmax(1).item()
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

env.close()

if frames:
    imageio.mimsave("./DemoResult/dqn_run.mp4", frames, fps=30)

print(f"Total reward episode: {total_reward:.2f}")
print("Video disimpan sebagai dqn_run.mp4")
