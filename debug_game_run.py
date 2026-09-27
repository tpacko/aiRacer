"""Run the real Game.run() loop headless (dummy pygame) for N seconds then stop."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import threading, time
from airacer.app import create_game

game = create_game(driver_name="neat", policy_path="artifacts_test/best_neat_policy.pkl")

# Stop Game.run() after wallclock seconds (sim time = wallclock here)
def stopper():
    time.sleep(22.0)  # ~2640 ticks at 120 Hz
    game.running = False

threading.Thread(target=stopper, daemon=True).start()
start = time.perf_counter()
game.run()
elapsed = time.perf_counter() - start

sim = game.simulation
print(
    f"wallclock={elapsed:.2f}s sim_steps={sim.steps} "
    f"progress_now={sim.observation.progress*100:.2f}% "
    f"max_progress={sim.max_progress*100:.2f}% "
    f"collisions={sim.collision_count} "
    f"forward_progress={game.env.forward_progress*100:.1f}%"
)
