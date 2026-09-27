"""Replay a saved NEAT policy in the exact training env and print metrics."""
from airacer.env import RaceEnv
from airacer.neat_policy import load_neat_policy
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "artifacts_test/best_neat_policy.pkl"
max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2400

policy = load_neat_policy(path)
env = RaceEnv(max_steps=max_steps)
obs, _ = env.reset()
total_reward = 0.0
terminated = truncated = False
info = {}
while not terminated and not truncated:
    action = policy.act(obs)
    step = env.step(action)
    obs = step.observation
    total_reward += step.reward
    terminated = step.terminated
    truncated = step.truncated
    info = step.info
mono = info.get("monotonic_progress", info["forward_progress"])
wm = info.get("max_watermark", info["forward_progress"])
cp = info.get("checkpoints_hit", 0)
col = info["collision_count"]
print(
    f"steps={info['step_count']} watermark={wm*100:.1f}% monotonic={mono*100:.1f}% "
    f"checkpoints={cp} col={col} fitness={total_reward:.2f} "
    f"terminated={terminated} truncated={truncated}"
)
