import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from wrappers import make_env

# Struktur data untuk menyimpan pengalaman
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Replay Buffer (menyimpan pengalaman agen)
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

# Konfigurasi QNetwork
class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=[512, 512, 256]):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Regularisasi agar tidak overfitting
            input_dim = h
        layers.append(nn.Linear(input_dim, n_actions))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# Inisiasi Agen DQN 
class DQNAgent:
    def __init__(self, env, device='cpu'):
        self.env = env
        self.device = device

        # Inisialisasi dimensi observasi dan jumlah aksi
        obs = np.array(env.reset(), dtype=np.float32)
        self.obs_dim = obs.shape[0]
        self.n_actions = env.action_space.n

        # Inisialisasi Q-Network dan Target Network
        self.q = QNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.q_target = QNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        # Optimizer dan hyperparameter utama
        self.optim = optim.Adam(self.q.parameters(), lr=1e-4)
        self.replay = ReplayBuffer(500000)
        self.gamma = 0.99
        self.batch_size = 128
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.995
        self.learn_start = 5000
        self.tau = 0.005
        self.step_count = 0

    # Memilih aksi berdasarkan epsilon-greedy policy
    def select_action(self, state):
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q(s)
            return int(qvals.argmax().item())

    # Menyimpan transisi ke replay buffer
    def push_transition(self, *args):
        self.replay.push(*args)

    # Proses training DQN
    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None
        batch = self.replay.sample(self.batch_size)
        states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Hitung Q(s, a) dan target Q
        q_values = self.q(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.q_target(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + (1.0 - dones) * self.gamma * q_next

        # Hitung dan update loss
        loss = nn.functional.mse_loss(q_values, q_target)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optim.step()

        # Soft update untuk target network
        for target_param, param in zip(self.q_target.parameters(), self.q.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()

    # Menyimpan model ke file
    def save(self, path):
        torch.save(self.q.state_dict(), path)

# Fungsi utama untuk training agen
def run_training(num_episodes=2000, max_steps_per_episode=1000):
    env = make_env(shaping=True, gym_old_api=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DQNAgent(env, device=device)
    scores = []

    # Loop utama training per episode
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        ep_reward = 0.0

        # Loop langkah per episode
        for t in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.push_transition(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward
            agent.step_count += 1

            # Latih agen jika buffer sudah cukup terisi
            if agent.step_count > agent.learn_start:
                agent.train_step()

            # Kurangi nilai epsilon untuk eksplorasi → eksploitasi
            agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
            if done:
                break

        scores.append(ep_reward)

        # Cetak hasil setiap 10 episode
        if ep % 10 == 0:
            avg_reward = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            print(f"Episode {ep:4d}  reward={ep_reward:.2f}  avg_last10={avg_reward:.2f}  eps={agent.eps:.3f}  replay={len(agent.replay)}")

        # Simpan model setiap 200 episode
        if ep % 200 == 0:
            agent.save(f"./DQN_PytorchFinal/dqn_pytorch_ep{ep}.pt")

    # Simpan model akhir
    agent.save("./DQN_PytorchFinal/dqn_pytorch_final.pt")
    print("Training completed.")

# Eksekusi script
if __name__ == "__main__":
    run_training(num_episodes=2000)
