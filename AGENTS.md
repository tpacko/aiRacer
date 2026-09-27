# aiRacer

`aiRacer` is a top-down 2D car racing simulator built as a foundation for AI experimentation.

The project starts with a manually drivable racing game controlled by keyboard input. The initial focus is on a clean, tunable driving model, a simple track, and a clear entrypoint for manual play and iteration.

The architecture should be designed from the start to support later AI work, including:
- interchangeable control sources (keyboard, scripted agents, neural networks, NEAT, genetic algorithms)
- hooks for control callbacks
- sensing of the car's surroundings
- deterministic simulation updates suitable for training and evaluation

Manual driving is the first milestone. AI training and experimentation layers will be added on top of the same simulation and control interfaces rather than bolted on separately later.

## Project overview

### Layout

- `airacer/` — core package
  - `simulation.py` — deterministic car/physics step
  - `track.py` — track geometry and queries
  - `game.py` — pygame loop, rendering, mode orchestration
  - `controllers.py` — keyboard input controller
  - `scripted.py` — controller that wraps a policy callable
  - `policies.py` — policy factory (scripted / learned policies)
  - `neat_policy.py` — NEAT training and policy loading
  - `learn.py` — learning-based training entrypoint
  - `env.py` — training environment wrapping the simulation
  - `config.py` — tunable constants
  - `app.py` — CLI entrypoint for interactive runs
- `main.py` — run the interactive game (default: keyboard)
- `scripted_main.py` — run the game with the scripted controller
- `neat_main.py` — train / evaluate a NEAT policy
- `learn_main.py` — run the learning entrypoint
- `debug_game.py`, `debug_game_run.py`, `verify_live.py`, `verify_neat.py` — ad-hoc debug/verification scripts
- `artifacts*/` — saved training artifacts
- `requirements.txt` — runtime deps: `pygame`, `numpy`, `neat-python`

### Running

Install deps and launch the interactive game:

```
pip install -r requirements.txt
python main.py                      # keyboard
python main.py --driver scripted    # scripted policy
python neat_main.py                 # NEAT training
python learn_main.py                # learning entrypoint
```

### Design notes

- Simulation is intended to step deterministically so the same control trace reproduces the same run — required for training and evaluation.
- Controllers and policies are decoupled: any control source conforming to the controller interface can drive the car, so keyboard, scripted, and learned agents share one pipeline.

### Logging preference

- Prefer the RL driver style for long-running training and verification: clear phase banners, explicit configuration at phase start, per-update progress, behavior metrics, optimizer/value diagnostics, artifact paths, and `flush=True` on important prints.
- Avoid silent long-running loops. If a phase can take noticeable time, log what started, what is being updated, where outputs are written, and enough rolling stats to understand whether learning is improving or failing.
