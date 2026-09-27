"""Run Game loop headless (dummy pygame video driver) and print progress."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import time
from airacer.app import create_game

game = create_game(driver_name="neat", policy_path="artifacts_test/best_neat_policy.pkl")
start = time.perf_counter()
ticks_target = 2400
dt = game.simulation_tick
# Drive the loop manually, deterministically — same inner work as Game.run
for i in range(ticks_target):
    game._sample_controller()
    game._simulate(dt)
print(
    f"ticks={ticks_target} "
    f"progress_now={game.simulation.observation.progress*100:.2f}% "
    f"collisions={game.simulation.collision_count} "
    f"max_progress={game.simulation.max_progress*100:.2f}%"
)
