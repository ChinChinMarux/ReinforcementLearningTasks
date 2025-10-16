import numpy as np
import gymnasium as gym

class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs = out
            info = {}
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        elif len(out) == 4:
            obs, reward, done, info = out
            terminated = done
            truncated = False
        else:
            raise RuntimeError("Unexpected env.step return signature")
        s = None
        try:
            state = getattr(self.env, "state", None)
            if state is not None:
                normalizer = getattr(self.env, "_normalizer", None)
                if callable(normalizer):
                    s = state * normalizer()
                else:
                    s = state
        except Exception:
            s = None
        if s is None:
            try:
                obs_arr = np.array(obs, dtype=np.float32)
                s = obs_arr
            except Exception:
                s = None
        x = y = vx = vy = sin_th = cos_th = omega = dx = dy = 0.0
        try:
            x = float(s[0])
            y = float(s[1])
            vx = float(s[2])
            vy = float(s[3])
            sin_th = float(s[4])
            cos_th = float(s[5])
            omega = float(s[6])
            dx = float(s[7])
            dy = float(s[8])
        except Exception:
            pass
        theta = float(np.arctan2(sin_th, cos_th))
        dist = float(np.sqrt(dx * dx + dy * dy))
        screen_h = getattr(self.env, "screen_h", 1.0)
        screen_w = getattr(self.env, "screen_w", 1.0)
        r_progress = -0.001 * dist  # Small penalty untuk distance
        r_alignment = 1.0 - (abs(theta) / np.pi)  # Reward untuk alignment
        r_velocity = 0.01 * (abs(vx) + abs(vy)) if dist < 200 else 0.0  # Reward velocity hanya saat dekat
        
        shaped = r_progress + r_alignment + r_velocity
        landed = False
        floor_y = getattr(self.env, "floor_y", None)
        if floor_y is not None:
            if y <= floor_y:
                landed = True
        tx, ty = getattr(self.env, "target_pos", (None, None))
        target_w = getattr(self.env, "target_w", 0.0)
        target_h = getattr(self.env, "target_h", 0.0)
        target_collide = False
        try:
            half_w, half_h = target_w / 2.0, target_h / 2.0
            if tx is not None:
                target_collide = (tx - half_w <= x <= tx + half_w) and (ty - half_h <= y <= ty + half_h)
        except Exception:
            target_collide = False
        if target_collide and abs(theta) < 0.35 and abs(vx) < 5.0 and abs(vy) < 5.0:
            shaped += 100.0  # Big positive reward
            terminated = True
        elif target_collide:
            shaped += 10.0   # Smaller positive reward
            terminated = True
        elif landed:
            shaped -= 5.0    # Reduced penalty
            terminated = True
            
        return obs, shaped, bool(terminated), bool(truncated), info

class GymnasiumToGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs = out
        return obs

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs, reward, done, info
        elif len(out) == 4:
            return out
        else:
            raise RuntimeError("Unexpected env.step return signature: len=%d" % len(out))

def make_env(shaping=True, gym_old_api=True):
    import rocket_env
    env = rocket_env.SimpleRocketEnv(render_mode=None)
    if shaping:
        env = RewardShapingWrapper(env)
    if gym_old_api:
        env = GymnasiumToGymWrapper(env)
    return env
