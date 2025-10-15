# wrappers.py
import numpy as np
import gymnasium as gym
import types

# ---------------------------
#  Reward shaping wrapper
# ---------------------------
class RewardShapingWrapper(gym.Wrapper):
    """
    Wrap the SimpleRocketEnv (Gymnasium) and replace the environment's reward
    with a shaped reward that encourages:
      - smaller distance to target
      - upright orientation (theta ~ 0)
      - low linear/angular velocity on landing
      - large positive reward for safe landing on target
    """
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # access denormalized state
        s = self.env.state * self.env._normalizer()
        x, y, vx, vy, sin_th, cos_th, omega, dx, dy = s
        theta = np.arctan2(sin_th, cos_th)
        distance = np.sqrt(dx*dx + dy*dy)

        # basic shaping:
        # - negative for distance (closer = better)
        # - penalty for tilt (abs(theta))
        # - penalty for high speeds near platform
        r_dist = -0.5 * distance / (self.env.screen_h)    # normalize roughly
        r_angle = -2.0 * (abs(theta) / np.pi)             # big penalty for tilt
        r_speed = -0.02 * (abs(vx) + abs(vy))
        r_omega = -0.1 * (abs(omega) / 20.0)

        # base shaped reward
        shaped = r_dist + r_angle + r_speed + r_omega

        # check landing/crash conditions from env to decide terminal reward
        # We rely on env flags: detect when termination occurred and why via info if available.
        # But env does not report cause, so we re-check geometry:
        landed = False
        if y <= self.env.floor_y:
            landed = True

        tx, ty = self.env.target_pos
        half_w, half_h = self.env.target_w/2, self.env.target_h/2
        target_collide = (tx-half_w <= x <= tx+half_w) and (ty-half_h <= y <= ty+half_h)

        # reward for successful, safe landing on target: upright and slow
        if target_collide and abs(theta) < 0.35 and abs(vx) < 5.0 and abs(vy) < 5.0:
            shaped += 200.0  # big positive reward for clean landing on target
        elif target_collide:
            shaped += 50.0   # landed but not ideal
        elif landed:
            shaped -= 50.0   # crashed on floor away from target

        # Keep reward bounded (optional)
        shaped = float(np.clip(shaped, -500.0, 500.0))

        # return in Gym (old) style: obs, reward, done, info
        done = bool(terminated or truncated)
        # But keep gymnasium style outputs for compatibility with training scripts that expect old gym:
        # We'll return obs, shaped, done, info (old gym style)
        return obs, shaped, done, info

# ---------------------------
#  Gymnasium -> Gym wrapper
# ---------------------------
class GymnasiumToGymWrapper(gym.Wrapper):
    """
    Convert Gymnasium API (reset -> obs, info) and step -> (obs, reward, terminated, truncated, info)
    into the old Gym API reset->obs and step->(obs, reward, done, info) expected by many libs.
    Use this AFTER RewardShapingWrapper if you want shaped rewards.
    """
    def __init__(self, env):
        # env expected to be a gymnasium.Env or a wrapper around it
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs  # old Gym returns only observation

    def step(self, action):
        out = self.env.step(action)
        # env.step may return either gymnasium or old gym style depending on inner wrappers.
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs, reward, done, info
        elif len(out) == 4:
            # already old-style
            return out
        else:
            raise RuntimeError("Unexpected env.step return signature: len=%d" % len(out))

# small helper to build env
def make_env(shaping=True, gym_old_api=True):
    import rocket_env
    env = rocket_env.SimpleRocketEnv(render_mode=None)
    if shaping:
        env = RewardShapingWrapper(env)
    if gym_old_api:
        env = GymnasiumToGymWrapper(env)
    return env
