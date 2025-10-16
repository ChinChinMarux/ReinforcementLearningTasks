import numpy as np
import gymnasium as gym

# Memberikan shaping reward tambahan agar agen belajar lebih cepat
class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    # Reset environment dan kembalikan observation (dengan info jika ada)
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs = out
            info = {}
        return obs, info

    # Step environment dengan perhitungan reward tambahan
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

        # Ambil nilai state aktual dari environment
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

        # Ekstrak variabel penting dari state (posisi, kecepatan, orientasi)
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

        # Hitung variabel tambahan untuk shaping reward
        theta = float(np.arctan2(sin_th, cos_th))
        dist = float(np.sqrt(dx * dx + dy * dy))
        screen_h = getattr(self.env, "screen_h", 1.0)
        screen_w = getattr(self.env, "screen_w", 1.0)

        # Komponen shaping reward
        r_progress = -0.001 * dist                # Penalti kecil untuk jarak jauh dari target
        r_alignment = 1.0 - (abs(theta) / np.pi)  # Reward untuk menjaga orientasi tegak
        r_velocity = 0.01 * (abs(vx) + abs(vy)) if dist < 200 else 0.0  # Reward kecil untuk kontrol kecepatan saat dekat target

        # Total reward shaping dasar
        shaped = r_progress + r_alignment + r_velocity

        # Deteksi kondisi mendarat
        landed = False
        floor_y = getattr(self.env, "floor_y", None)
        if floor_y is not None and y <= floor_y:
            landed = True

        # Dapatkan posisi target dari environment
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

        # Tambahkan reward terminal (sukses / gagal mendarat)
        if target_collide and abs(theta) < 0.35 and abs(vx) < 5.0 and abs(vy) < 5.0:
            shaped += 100.0   # Reward besar untuk pendaratan sempurna
            terminated = True
        elif target_collide:
            shaped += 10.0    # Reward kecil jika menyentuh target tapi belum ideal
            terminated = True
        elif landed:
            shaped -= 5.0     # Penalti kecil jika jatuh di luar target
            terminated = True

        return obs, shaped, bool(terminated), bool(truncated), info

# Mengubah API Gymnasium (baru) ke format Gym (lama)
class GymnasiumToGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    # Reset environment dengan format observasi tunggal
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs = out
        return obs

    # Step environment dan ubah keluaran ke format (obs, reward, done, info)
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
