import pygame
from stable_baselines3 import PPO
from mario_env import MarioEnv

env = MarioEnv(render_mode=True)
model = PPO.load("mario_model")

obs = env.reset()
done = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    env.render()

    if done:
        obs = env.reset()