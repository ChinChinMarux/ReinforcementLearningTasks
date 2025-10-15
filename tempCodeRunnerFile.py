from rocket_env import SimpleRocketEnv
import pygame

if __name__ == "__main__":

    env = SimpleRocketEnv()
    state, _ = env.reset()
    done = False

    while not done:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1   # main thruster
                elif event.key == pygame.K_RIGHT:
                    action = 2   # right-side thruster
                elif event.key == pygame.K_LEFT:
                    action = 3   # left-side thruster

        state, reward, done, truncated, info = env.step(action)
        print(state)
        env.render()