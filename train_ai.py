from stable_baselines3 import PPO
from mario_env import MarioEnv

# Create environment (no rendering during training)
env = MarioEnv(render_mode=False)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=4096,       # longer rollouts help with jump timing credit
    batch_size=128,
    gamma=0.99,
    ent_coef=0.02,      # higher entropy — more exploration of new actions
    vf_coef=0.5,
    clip_range=0.2,
)
model.learn(total_timesteps=3_000_000)

# Save model
model.save("mario_model")

print("Training complete. Model saved as mario_model.zip")