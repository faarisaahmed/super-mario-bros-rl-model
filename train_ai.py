from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mario_env import MarioEnv


def make_env():
    return MarioEnv(render_mode=False)


# ── Environment ───────────────────────────────────────────────────────────────
env = DummyVecEnv([make_env])

# VecNormalize tracks a running mean/std of observations AND rewards,
# normalising both automatically. This is the single biggest fix for
# the noisy value_loss — the critic no longer sees reward spikes of
# wildly different magnitudes each update.
env = VecNormalize(
    env,
    norm_obs=True,       # normalise observations to ~N(0,1)
    norm_reward=True,    # normalise rewards to ~N(0,1)
    clip_obs=10.0,       # clip normalised obs to [-10, 10]
    clip_reward=10.0,    # clip normalised reward to [-10, 10]
    gamma=0.95,          # must match model gamma
)

# ── Callbacks ─────────────────────────────────────────────────────────────────
# Save a checkpoint every 100k steps so a crash never loses everything
checkpoint_cb = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="mario",
    save_vecnormalize=True,   # also saves the normalisation stats
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,

    # Lower learning rate prevents the spike-and-crash you saw at 1.5M
    learning_rate=0.0001,

    # Longer rollouts give the agent more context per update,
    # important when rewards are sparse (gaps are infrequent)
    n_steps=4096,
    batch_size=128,

    # How many times to reuse each rollout — 10 epochs squeezes more
    # signal from each batch without collecting new experience
    n_epochs=10,

    # Shorter horizon (0.95 vs 0.99) makes the critic's job easier —
    # it doesn't have to predict as far into the future
    gamma=0.95,

    # GAE lambda: how much to rely on bootstrapped vs actual returns
    gae_lambda=0.95,

    # Higher entropy coefficient keeps the agent exploring all 6 actions
    # rather than collapsing to "always walk right"
    ent_coef=0.02,

    # Increased from 0.5 — tells PPO to prioritise fixing value loss,
    # which was the main training problem shown in the graphs
    vf_coef=1.0,

    # Max gradient norm — prevents any single bad batch from
    # causing a huge destructive policy update
    max_grad_norm=0.5,

    clip_range=0.2,

    tensorboard_log="./tensorboard/",
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Starting training...")
print("Monitor progress: tensorboard --logdir ./tensorboard/")
print("Checkpoints saved to: ./checkpoints/")
print()

model.learn(
    total_timesteps=5_000_000,
    callback=checkpoint_cb,
    progress_bar=True,
)

# ── Save ──────────────────────────────────────────────────────────────────────
model.save("mario_model")
env.save("mario_vecnormalize.pkl")   # IMPORTANT: save normalisation stats

print()
print("Training complete.")
print("Model saved:       mario_model.zip")
print("Norm stats saved:  mario_vecnormalize.pkl")
print()
print("To resume from a checkpoint:")
print("  model = PPO.load('checkpoints/mario_XXXXXXX_steps')")
print("  env = VecNormalize.load('mario_vecnormalize.pkl', DummyVecEnv([make_env]))")