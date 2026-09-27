"""Mimic Game's fixed-timestep loop without pygame, using ScriptedController."""
import sys
from airacer.neat_policy import load_neat_policy
from airacer.scripted import ScriptedController
from airacer.simulation import Simulation
from airacer import config

path = sys.argv[1] if len(sys.argv) > 1 else "artifacts_test/best_neat_policy.pkl"
ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 2400

policy = load_neat_policy(path)
controller = ScriptedController(policy)
sim = Simulation()

dt = 1.0 / config.SIMULATION_HZ
max_progress_cumulative = 0.0
prev_progress = sim.observation.progress
forward_progress = 0.0
for i in range(ticks):
    command = controller.sample(sim)
    sim.step(command.control, dt)
    p = sim.observation.progress
    delta = p - prev_progress
    if delta < -0.5:
        delta += 1.0
    elif delta > 0.5:
        delta -= 1.0
    forward_progress += max(delta, 0.0)
    prev_progress = p

print(
    f"ticks={ticks} dist={forward_progress*100:.1f}% "
    f"progress_now={sim.observation.progress*100:.2f}% "
    f"collisions={sim.collision_count}"
)
