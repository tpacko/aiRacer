# aiRacer Plan

## Milestone 1: Project scaffold
- Create a minimal package structure for simulation, rendering, input, track, and controllers.
- Add a single runnable entrypoint for local play.
- Define core config constants so tuning does not get buried in logic.

Checkpoint:
- Running the app opens a window and starts a fixed-timestep game loop.

## Milestone 2: Simulation core
- Implement a deterministic top-down simulation loop separate from rendering.
- Add `CarState` and `ControlInput` data models.
- Ensure the simulation updates from control values, not directly from keyboard state.

Checkpoint:
- The car state updates correctly in a headless step loop.

## Milestone 3: Car dynamics
- Implement the initial top-down driving model: throttle, brake/reverse, steering, drag, lateral grip, angular rotation.
- Keep the model simple and tunable rather than physically complete.
- Expose parameters like acceleration, max speed, turn rate, friction, and grip.

Checkpoint:
- A car can move, turn, coast, and stop in open space with stable behavior.

## Milestone 4: Keyboard controller
- Add a `KeyboardController` that maps keys to `ControlInput`.
- Route manual input through the same controller interface future AI agents will use.
- Support at least throttle, brake/reverse, steer left, steer right, reset.

Checkpoint:
- You can drive the car manually around an empty scene using the keyboard.

## Milestone 5: Camera and rendering
- Render the car clearly from a top-down view.
- Add a basic background or grid and heading visualization for debugging.
- Keep debug overlays easy to toggle.

Checkpoint:
- Manual driving feels visually readable and responsive.

## Milestone 6: Track system
- Define a first track format, likely simple polygons or wall segments for inner and outer boundaries.
- Add spawn point and initial heading.
- Render the track and constrain driving to a raceable layout.

Checkpoint:
- The game starts with the car positioned on a simple track.

## Milestone 7: Collision handling
- Detect car collisions against track boundaries.
- Start with a simple collision model that is robust and easy to debug.
- Decide first response behavior: stop, bounce lightly, or mark crashed and reset.

Checkpoint:
- Driving into walls consistently triggers the intended collision behavior.

## Milestone 8: Driving feel tuning
- Tune acceleration, turn response, drift, grip, and collision response until keyboard driving is satisfying.
- Add a quick reset or restart flow to speed up iteration.
- Keep parameters centralized for fast experiments.

Checkpoint:
- The car is manually drivable in a way that feels predictable and controllable on the track.

## Milestone 9: Progress tracking
- Add checkpoints or centerline progress measurement.
- Track lap progress, current checkpoint, lap completion, and off-track or crash state.
- Make progress available from the simulation state, not only the UI.

Checkpoint:
- The simulator can measure whether the driver is progressing around the course.

## Milestone 10: Sensor system
- Implement ray-based sensing from the car to nearby walls.
- Return normalized distances and optionally draw debug rays on screen.
- Include a compact observation object that can later feed AI agents.

Checkpoint:
- The car exposes sensor readings every frame, and they match the visible scene.

## Milestone 11: Controller abstraction for AI
- Finalize a clean controller interface used by keyboard and future AI agents.
- Ensure the simulation accepts controller output only.
- Keep observation and action shapes explicit and stable.

Checkpoint:
- Swapping `KeyboardController` for another controller requires no simulation rewrite.

## Milestone 12: AI-ready environment boundary
- Add a simple environment wrapper with `reset`, `step`, observation, termination, and reward or progress hooks.
- Support headless stepping for later training runs.
- Keep rendering optional.

Checkpoint:
- The same simulator can run either interactively or as an experiment environment.
