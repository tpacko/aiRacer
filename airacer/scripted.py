"""Scripted policy adapters and baseline episode runner."""

from __future__ import annotations

from dataclasses import dataclass

from .controllers import ControllerCommand
from .env import RaceEnv, observation_to_vector
from .policies import DriverPolicy, HeuristicPolicy
from .simulation import ControlInput, Simulation


@dataclass
class ScriptedEpisodeResult:
    steps: int
    total_reward: float
    progress: float
    best_progress: float
    collisions: int
    terminated: bool
    truncated: bool


class ScriptedController:
    """Adapter that lets a policy control the live pygame game."""

    def __init__(self, driver: DriverPolicy | None = None):
        self.driver = driver or HeuristicPolicy()
        self.policy_metadata: dict | None = getattr(self.driver, "metadata", None)

    def sample(self, simulation: Simulation):
        observation = observation_to_vector(simulation.observation)
        steer, acceleration = self.driver.act(observation)
        return ControllerCommand(
            control=ControlInput(steer=steer, acceleration=acceleration),
            reset_requested=False,
        )


def run_scripted_episode(
    env: RaceEnv | None = None,
    max_steps: int | None = None,
    driver: DriverPolicy | None = None,
):
    env = env or RaceEnv(max_steps=max_steps, fast_observations=True)
    if max_steps is not None:
        env.max_steps = max_steps

    driver = driver or HeuristicPolicy()
    observation, info = env.reset()
    total_reward = 0.0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = driver.act(observation)
        step_result = env.step(action)
        observation = step_result.observation
        total_reward += step_result.reward
        terminated = step_result.terminated
        truncated = step_result.truncated
        info = step_result.info

    return ScriptedEpisodeResult(
        steps=info["step_count"],
        total_reward=total_reward,
        progress=info["progress"],
        best_progress=info["best_progress"],
        collisions=info["collision_count"],
        terminated=terminated,
        truncated=truncated,
    )
