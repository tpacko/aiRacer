"""AI-ready environment wrapper around the racing simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

from . import config
from .simulation import ControlInput, Observation, Simulation
from .track import DEFAULT_TRACK_NAME


@dataclass
class StepResult:
    observation: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict


def observation_to_vector(observation: Observation) -> list[float]:
    """Canonical RL observation vector. Shared between training (RaceEnv)
    and inference (ScriptedController) so a policy is fed the exact same
    features in both contexts.
    """
    if observation.vector is not None:
        return observation.vector
    sensor_values = [reading.normalized_distance for reading in observation.sensor_readings]
    angular_velocity_norm = observation.angular_velocity / max(
        config.CAR_MAX_ANGULAR_VELOCITY, 1e-6
    )
    lookahead_values = [
        sample.heading_delta / math.pi for sample in observation.lookahead_samples
    ]
    return [
        observation.centerline_error / config.SENSOR_RAY_LENGTH,
        observation.heading_error / math.pi,
        observation.speed / config.CAR_MAX_SPEED,
        observation.forward_speed / config.CAR_MAX_SPEED,
        *sensor_values,
        observation.lateral_speed / config.CAR_MAX_SPEED,
        angular_velocity_norm,
        *lookahead_values,
    ]


class RaceEnv:
    """Thin environment wrapper suitable for scripted agents and training loops."""

    def __init__(
        self,
        simulation_hz: int | None = None,
        max_steps: int | None = None,
        track_name: str = DEFAULT_TRACK_NAME,
        fast_observations: bool = False,
    ):
        self.simulation_hz = simulation_hz or config.SIMULATION_HZ
        self.dt = 1.0 / self.simulation_hz
        self.max_steps = max_steps or config.ENV_MAX_STEPS
        self.fast_observations = fast_observations
        self.simulation = Simulation(
            track_name=track_name,
            compact_observations=fast_observations,
        )
        self.step_count = 0
        self.last_progress = 0.0
        self.monotonic_progress = 0.0
        self.max_watermark = 0.0
        self.checkpoints_hit = 0
        self.completed_laps = 0
        self.last_collision_count = 0
        self.low_speed_steps = 0
        self.lap_completed = False
        self.prev_steer = 0.0
        self.lap_start_step = 0

    def reset(self, *, randomize_start: bool = False, rng: random.Random | None = None):
        self.simulation.reset(randomize_start=randomize_start, rng=rng)
        self.step_count = 0
        self.last_progress = self.simulation.observation.progress
        self.monotonic_progress = 0.0
        self.max_watermark = 0.0
        self.checkpoints_hit = 0
        self.completed_laps = 0
        self.last_collision_count = self.simulation.observation.collision_count
        self.low_speed_steps = 0
        self.lap_completed = False
        self.prev_steer = 0.0
        self.lap_start_step = 0
        observation = self._observation_vector(self.simulation.observation)
        info = self._info(progress_delta=0.0, collision_delta=0, lap_completed=False)
        return observation, info

    def step(self, action: Sequence[float]):
        control = self._action_to_control(action)
        snapshot = self.simulation.step(control, self.dt)
        observation = snapshot.observation
        self.step_count += 1
        steer_delta = abs(control.steer - self.prev_steer)
        self.prev_steer = control.steer

        progress_delta = self._progress_delta(self.last_progress, observation.progress)
        self.monotonic_progress += progress_delta

        watermark_advance = max(0.0, self.monotonic_progress - self.max_watermark)
        if watermark_advance > 0.0:
            self.max_watermark = self.monotonic_progress

        new_checkpoints = int(self.max_watermark * config.ENV_CHECKPOINTS_PER_LAP)
        checkpoint_delta = max(0, new_checkpoints - self.checkpoints_hit)
        prior_laps = self.checkpoints_hit // config.ENV_CHECKPOINTS_PER_LAP
        self.checkpoints_hit = new_checkpoints
        new_laps = self.checkpoints_hit // config.ENV_CHECKPOINTS_PER_LAP
        lap_count_delta = new_laps - prior_laps
        lap_time_bonus = 0.0
        if lap_count_delta > 0:
            self.completed_laps = new_laps
            self.lap_completed = True
            steps_in_lap = max(1, self.step_count - self.lap_start_step)
            remaining_fraction = max(0.0, 1.0 - steps_in_lap / max(1, self.max_steps))
            lap_time_bonus = (
                lap_count_delta * remaining_fraction * config.ENV_LAP_TIME_BONUS_SCALE
            )
            self.lap_start_step = self.step_count

        collision_delta = observation.collision_count - self.last_collision_count

        reward = self._reward(
            observation=observation,
            progress_delta=progress_delta,
            watermark_advance=watermark_advance,
            checkpoint_delta=checkpoint_delta,
            collision_delta=collision_delta,
            lap_count_delta=lap_count_delta,
            lap_time_bonus=lap_time_bonus,
            steer_delta=steer_delta,
        )
        terminated = collision_delta > 0
        truncated = self.step_count >= self.max_steps or self._is_stuck(observation, progress_delta)

        info = self._info(progress_delta, collision_delta, lap_count_delta > 0)
        self.last_progress = observation.progress
        self.last_collision_count = observation.collision_count

        return StepResult(
            observation=self._observation_vector(observation),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def step_training(self, action: Sequence[float]):
        """Lower-allocation step path used by PPO rollout collection.

        It avoids creating StepResult and per-step info dictionaries. The full
        game path still uses ``step`` so rendering keeps sensor hit-points and
        lookahead samples.
        """
        control = self._action_to_control(action)
        observation = self.simulation.step_observation(control, self.dt)
        self.step_count += 1
        steer_delta = abs(control.steer - self.prev_steer)
        self.prev_steer = control.steer

        progress_delta = self._progress_delta(self.last_progress, observation.progress)
        self.monotonic_progress += progress_delta

        watermark_advance = max(0.0, self.monotonic_progress - self.max_watermark)
        if watermark_advance > 0.0:
            self.max_watermark = self.monotonic_progress

        new_checkpoints = int(self.max_watermark * config.ENV_CHECKPOINTS_PER_LAP)
        checkpoint_delta = max(0, new_checkpoints - self.checkpoints_hit)
        prior_laps = self.checkpoints_hit // config.ENV_CHECKPOINTS_PER_LAP
        self.checkpoints_hit = new_checkpoints
        new_laps = self.checkpoints_hit // config.ENV_CHECKPOINTS_PER_LAP
        lap_count_delta = new_laps - prior_laps
        lap_time_bonus = 0.0
        if lap_count_delta > 0:
            self.completed_laps = new_laps
            self.lap_completed = True
            steps_in_lap = max(1, self.step_count - self.lap_start_step)
            remaining_fraction = max(0.0, 1.0 - steps_in_lap / max(1, self.max_steps))
            lap_time_bonus = (
                lap_count_delta * remaining_fraction * config.ENV_LAP_TIME_BONUS_SCALE
            )
            self.lap_start_step = self.step_count

        collision_delta = observation.collision_count - self.last_collision_count
        reward = self._reward(
            observation=observation,
            progress_delta=progress_delta,
            watermark_advance=watermark_advance,
            checkpoint_delta=checkpoint_delta,
            collision_delta=collision_delta,
            lap_count_delta=lap_count_delta,
            lap_time_bonus=lap_time_bonus,
            steer_delta=steer_delta,
        )
        terminated = collision_delta > 0
        truncated = self.step_count >= self.max_steps or self._is_stuck(observation, progress_delta)

        self.last_progress = observation.progress
        self.last_collision_count = observation.collision_count

        return (
            self._observation_vector(observation),
            reward,
            terminated,
            truncated,
            progress_delta,
            collision_delta,
            lap_count_delta > 0,
        )

    def load_track(self, track_name: str, *, randomize_start: bool = False, rng: random.Random | None = None):
        self.simulation.load_track(track_name, randomize_start=randomize_start, rng=rng)
        self.step_count = 0
        self.last_progress = self.simulation.observation.progress
        self.monotonic_progress = 0.0
        self.max_watermark = 0.0
        self.checkpoints_hit = 0
        self.completed_laps = 0
        self.last_collision_count = self.simulation.observation.collision_count
        self.low_speed_steps = 0
        self.lap_completed = False
        self.prev_steer = 0.0
        self.lap_start_step = 0
        observation = self._observation_vector(self.simulation.observation)
        info = self._info(progress_delta=0.0, collision_delta=0, lap_completed=False)
        return observation, info

    def sample_observation(self):
        return self._observation_vector(self.simulation.observation)

    def _action_to_control(self, action: Sequence[float]):
        if len(action) < 2:
            raise ValueError("Action must contain steer and acceleration values.")
        steer = _clamp(float(action[0]), -1.0, 1.0)
        acceleration = _clamp(float(action[1]), -1.0, 1.0)
        return ControlInput(steer=steer, acceleration=acceleration)

    def _observation_vector(self, observation: Observation):
        return observation_to_vector(observation)

    def _reward(
        self,
        observation: Observation,
        progress_delta: float,
        watermark_advance: float,
        checkpoint_delta: int,
        collision_delta: int,
        lap_count_delta: int,
        lap_time_bonus: float = 0.0,
        steer_delta: float = 0.0,
    ):
        reward = 0.0
        reward += progress_delta * config.ENV_PROGRESS_SHAPE_SCALE
        reward += watermark_advance * config.ENV_WATERMARK_REWARD_SCALE
        reward += checkpoint_delta * config.ENV_CHECKPOINT_BONUS
        reward += lap_count_delta * config.ENV_LAP_BONUS
        reward += lap_time_bonus
        reward -= max(-progress_delta, 0.0) * config.ENV_WRONG_WAY_PROGRESS_PENALTY_SCALE
        reward += max(observation.forward_speed, 0.0) * config.ENV_FORWARD_SPEED_REWARD_SCALE
        reward -= max(-observation.forward_speed, 0.0) * config.ENV_BACKWARD_SPEED_PENALTY_SCALE
        reward -= observation.centerline_error * config.ENV_CENTERLINE_PENALTY_SCALE
        reward -= abs(observation.heading_error) * config.ENV_HEADING_PENALTY_SCALE
        reward -= steer_delta * config.ENV_STEER_JERK_PENALTY_SCALE
        reward -= abs(observation.lateral_speed) * config.ENV_LATERAL_SLIP_PENALTY_SCALE
        if collision_delta > 0:
            reward -= collision_delta * config.ENV_COLLISION_PENALTY
        else:
            reward += config.ENV_SURVIVAL_BONUS
        return reward

    def _progress_delta(self, previous: float, current: float):
        delta = current - previous
        if delta < -0.5:
            return delta + 1.0
        if delta > 0.5:
            return delta - 1.0
        return delta

    def _is_stuck(self, observation: Observation, progress_delta: float):
        if observation.forward_speed < config.ENV_STUCK_SPEED_THRESHOLD and progress_delta < 0.0005:
            self.low_speed_steps += 1
        else:
            self.low_speed_steps = 0
        return self.low_speed_steps >= config.ENV_STUCK_WINDOW

    def _info(self, progress_delta: float, collision_delta: int, lap_completed: bool):
        observation = self.simulation.observation
        return {
            "step_count": self.step_count,
            "progress": observation.progress,
            "forward_progress": self.max_watermark,
            "monotonic_progress": self.monotonic_progress,
            "max_watermark": self.max_watermark,
            "checkpoints_hit": self.checkpoints_hit,
            "best_progress": self.max_watermark,
            "completed_laps": self.completed_laps,
            "progress_delta": progress_delta,
            "centerline_error": observation.centerline_error,
            "heading_error": observation.heading_error,
            "speed": observation.speed,
            "forward_speed": observation.forward_speed,
            "collision_count": observation.collision_count,
            "collision_delta": collision_delta,
            "lap_completed": lap_completed,
        }


def _clamp(value: float, minimum: float, maximum: float):
    return max(minimum, min(maximum, value))
