import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import random

class SimpleRocketEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode='human'):
        super().__init__()
        self.render_mode = render_mode
        # Simulation params
        self.dt, self.m, self.g = 0.025, 3.0, 9.81
        self.F_main, self.F_side = 400.0, 200.0

        # Screen & world
        self.screen_w, self.screen_h = 960, 480
        self.floor_y = 10.0

        # Rocket geometry & inertia
        self.w, self.h = 30.0, 60.0
        self.I = (1/12) * self.m * (self.w**2 + self.h**2)

        # Damping 
        self.b_linear  = 0.1   # linear drag coefficient
        self.b_angular = 0.05  # angular drag coefficient

        # Target rectangle center + size
        tx_init = int(random.random() * 500)
        self.target_pos = np.array([300.0 + tx_init, 40.0], np.float32)
        self.target_w, self.target_h = 240.0, 80.0
        self.target_min_x = 300.0
        self.target_max_x = 800.0
        self.target_vx    =  50.0   # units per second, adjust as needed

        # Launch pad center in world coords and dim
        self.launch_pad_pos = np.array([100.0, 50.0], dtype=np.float32)
        self.pad_w, self.pad_h = 40.0, 40.0

        # Observation: x, y, vx, vy, sinθ, cosθ, ω, dx, dy
        high = np.array([
            self.screen_w, self.screen_h,
            np.finfo(np.float32).max, np.finfo(np.float32).max,
            1., 1., #np.pi, 
            np.finfo(np.float32).max,
            self.screen_w, self.screen_h
        ], np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space      = spaces.Discrete(4)

        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            # Pygame & assets
            self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
            pygame.display.set_caption("Simple Rocket Env")
            self.clock = pygame.time.Clock()
            self.font  = pygame.font.SysFont("Arial", 16)

            base = os.path.dirname(__file__)
            load = lambda fn: pygame.image.load(os.path.join(base, fn)).convert_alpha()
            self.rocket_img = pygame.transform.scale(load("rocket.png"), (int(self.w), int(self.h)))
            self.target_img = pygame.transform.scale(load("target.png"),
                                                    (int(self.target_w), int(self.target_h)))

        self.state = None
        self.last_action = 0
        self.max_steps = 800          # maximum steps per episode
        self.step_count = 0           # initialize counter

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Start upright, zero velocity
        self.state = np.array([100., 50., 0., 60., 0., 1., 0., 0., 0.], np.float32) / self._normalizer()
        self.target_vx *= random.choice([1, -1])
        self.last_action = 0
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        self.step_count += 1

        x, y, vx, vy, sinθ, cosθ, ω, dx, dy = self.state * self._normalizer()
        Fx = Fy = torque = 0.0
        θ = np.arctan2(sinθ, cosθ)

        # Main or side thrusters
        if action == 1:
            Fx = self.F_main * np.sin(θ)
            Fy = self.F_main * np.cos(θ)
        elif action in (2, 3):
            s = 1 if action == 2 else -1
            torque = s * self.F_side * (self.h/2)
            Fx += -s * self.F_side * np.cos(θ)
            Fy += -s * self.F_side * np.sin(θ)

        # add linear drag: F_drag = -b_linear * v
        Fx += -self.b_linear * vx
        Fy += -self.b_linear * vy

        # add angular damping torque: τ_damp = -b_angular * ω
        torque += -self.b_angular * ω

        # integrate translational motion
        vy += ((Fy/self.m) - self.g) * self.dt
        vx += (Fx/self.m) * self.dt
        y  += vy * self.dt
        x  += vx * self.dt

        # integrate rotational motion
        ω += (torque / self.I) * self.dt
        θ += ω * self.dt
        sinθ, cosθ = np.sin(θ), np.cos(θ)

        # wrap angle and clip velocities as before…
        θ = (θ + np.pi) % (2*np.pi) - np.pi
        vx, vy = np.clip([vx, vy], -120, 120)
        ω = np.clip(ω, -20, 20)

        # target position
        tx, ty = self.target_pos
        dx, dy = x - tx, y - ty

        # moving target
        tvx = random.random() * self.target_vx
        tx = self.target_pos[0] + tvx * self.dt
        
        # bounce at bounds
        if tx < self.target_min_x:
            tx = self.target_min_x
            self.target_vx = -self.target_vx
        elif tx > self.target_max_x:
            tx = self.target_max_x
            self.target_vx = -self.target_vx
        self.target_pos[0] = tx

        # recompute relative distance
        dx = x - self.target_pos[0]
        dy = y - self.target_pos[1]

        # Floor collision
        landed = False
        if y <= self.floor_y:
            y, vy = self.floor_y, 0.0
            landed = True

        # Target collision
        tx, ty = self.target_pos
        half_w, half_h = self.target_w/2, self.target_h/2
        target_collide = tx-half_w<=x<=tx+half_w and ty-half_h<=y<=ty+half_h

        # Reward & done
        terminated, truncated, reward = False, False, 0.0
        if target_collide:
            terminated, reward = True, -1.0
        elif x<0 or x>self.screen_w or y>self.screen_h:
            terminated, reward = True, -1.0
        elif landed:
            terminated, reward = True, -1.0
        else:
            pass

        # check for max-step termination
        if self.step_count >= self.max_steps:
            truncated = True

        self.state = np.array([x, y, vx, vy, sinθ, cosθ, ω, dx, dy], np.float32) / self._normalizer()
        self.last_action = action

        return self.state, reward, terminated, truncated, {}
    
    def _normalizer(self):
        return np.array([self.screen_w, self.screen_h, 120., 120., 1., 1., 20., self.screen_w, self.screen_h])

    def render(self, mode='human'):
        if mode != 'human':
            return
        x, y, vx, vy, sinθ, cosθ, ω, dx, dy = self.state * self._normalizer()
        self.screen.fill((0,0,0))
        y_flip = self.screen_h - y
        θ = np.arctan2(sinθ, cosθ)

        # Draw rocket sprite rotated to match physics angle
        deg = -np.degrees(θ)
        rocket = pygame.transform.rotate(self.rocket_img, deg)
        rect = rocket.get_rect(center=(x, y_flip))
        self.screen.blit(rocket, rect.topleft)

        # Draw thrust flame
        if self.last_action == 1:
            flame = [(-5,self.h/2), (5,self.h/2), (0,self.h/2+20)]
        elif self.last_action in (2,3):
            s = -1 if self.last_action==2 else 1
            flame = [(s*(self.w/2+5),0), (s*self.w/2,-5), (s*self.w/2,5)]
        else:
            flame = []
        if flame:
            c, s_ = np.cos(θ), np.sin(θ)
            pts = [(int(x + c*px - s_*py), int(y_flip + s_*px + c*py))
                   for px,py in flame]
            pygame.draw.polygon(self.screen, (255,140,70), pts)

        # Floor & target
        fy = self.screen_h - self.floor_y
        pygame.draw.line(self.screen,(200,200,200),(0,fy),(self.screen_w,fy),2)

        tx, ty = self.target_pos
        tx, ty = tx-self.target_w/2, (self.screen_h-ty)-self.target_h/2
        self.screen.blit(self.target_img, (int(tx), int(ty)))

        # Pad
        px, py = self.launch_pad_pos
        screen_py = self.screen_h - py
        pad_rect = pygame.Rect(
            int(px - self.pad_w/2),      # left
            int(screen_py),              # top
            int(self.pad_w),             # width
            int(self.pad_h)              # height
        )
        pygame.draw.rect(self.screen, (80, 120, 80), pad_rect)

        # HUD
        info = f"x={x:.1f} y={y:.1f} vx={vx:.1f} vy={vy:.1f} θ={θ:.2f} ω={ω:.2f} dx={dx:.1f} dy={dy:.1f}"
        self.screen.blit(self.font.render(info, True, (255,255,255)), (10,10))

        pygame.display.flip()
        self.clock.tick(50)

    def close(self):
        if self.render_mode == 'human':
            pygame.quit()

