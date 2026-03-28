from stable_baselines3 import PPO
from mario_env import MarioEnv

env = MarioEnv(render_mode=False)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=4096,
    batch_size=128,
    gamma=0.99,
    ent_coef=0.02,
    vf_coef=0.5,
    clip_range=0.2,
    tensorboard_log="./logs/"   # 👈 ADD THIS
)

model.learn(total_timesteps=3_000_000)

model.save("mario_model")