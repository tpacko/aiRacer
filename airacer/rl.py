"""From-scratch PPO trainer for the aiRacer simulation.

Design goals:
  * readable, debuggable single-file PPO (no stable-baselines3)
  * CPU-only PyTorch
  * uses RaceEnv directly (no gymnasium wrapper)
  * curriculum: warmup -> generalize -> race_fast (see make_default_stages)
  * checkpoint via torch.save() for resume; policy export as MLPPolicy-compatible JSON
  * detailed per-rollout stats showing emergent driving behavior
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from . import config
from .env import RaceEnv
from .track import (
    DEFAULT_TRACK_NAME,
    advanced_track_names,
    available_track_names,
    basic_track_names,
    intermediate_track_names,
)


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------


@dataclass
class StageConfig:
    """One stage of the training curriculum.

    Stages are applied in order; each runs for `steps` policy steps (frame-skipped
    env transitions). A stage controls which tracks are sampled, whether episodes
    start randomly, the extra forward-speed reward shaping, and the PPO entropy
    bonus (higher = more exploration).
    """

    name: str
    steps: int
    track_names: List[str]
    randomize_start: bool
    forward_speed_reward_scale: float
    entropy_coef: float
    # Per-stage collision penalty. Higher in early stages to force the policy
    # to learn clean driving before it's rewarded for going fast (in later
    # stages the penalty relaxes so contact can be traded for pace).
    collision_penalty: float = config.ENV_COLLISION_PENALTY
    # Per-stage lap-time bonus. Lower than the original 400 so a fast lap
    # bonus can't paper over a crash.
    lap_time_bonus_scale: float = config.ENV_LAP_TIME_BONUS_SCALE
    # Small "stay near the centerline" shaping term (penalty per pixel of
    # lateral error per step). Larger in basics to teach clean lines, decays
    # to 0 in ``race_fast`` so the policy can pick its own racing line.
    centerline_penalty_scale: float = config.ENV_CENTERLINE_PENALTY_SCALE


@dataclass
class RLConfig:
    # network
    hidden_size: int = 128
    initial_log_std: float = -0.5  # sigma ~ 0.6 at start
    # rollout / PPO
    total_steps: int = 3_000_000
    rollout_steps: int = 2048
    rollout_workers: int = 1
    frame_skip: int = 4
    n_epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: Optional[float] = 100.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    # env
    max_steps_per_episode: int = 2400
    # curriculum
    stages: List[StageConfig] = field(default_factory=list)
    # artifacts
    checkpoint_path: str = "artifacts/ppo_state.pt"
    policy_json_path: str = "artifacts/ppo_policy.json"
    ppo_log_path: str = "artifacts/ppo_training_log.jsonl"
    # logging
    print_every_updates: int = 1
    recent_episode_window: int = 20  # rolling window for "recent" stats
    # critic prelearning used after actor-only warm-starts
    critic_pretrain_updates: int = 20
    critic_pretrain_epochs: int = 6
    # repro
    seed: int = 42


# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------


class Actor(nn.Module):
    """Gaussian policy: mean from a 1-hidden-layer tanh MLP, per-dim learnable log_std.

    The architecture (1 hidden tanh layer, tanh on output) matches MLPPolicy
    in airacer.policies exactly, so trained weights can be exported as JSON and
    loaded by the existing --driver policy runner.
    """

    def __init__(self, obs_dim: int, hidden_size: int, initial_log_std: float):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 2)
        self.log_std = nn.Parameter(torch.full((2,), float(initial_log_std)))
        nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc_out.weight, gain=0.01)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, obs: torch.Tensor):
        hidden = torch.tanh(self.fc1(obs))
        mean = torch.tanh(self.fc_out(hidden))
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def dist(self, obs: torch.Tensor) -> Normal:
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def export_mlp_dict(self) -> dict:
        return {
            "type": "mlp",
            "w1": self.fc1.weight.detach().cpu().tolist(),
            "b1": self.fc1.bias.detach().cpu().tolist(),
            "w2": self.fc_out.weight.detach().cpu().tolist(),
            "b2": self.fc_out.bias.detach().cpu().tolist(),
        }


class Critic(nn.Module):
    """Value function: 2-hidden-layer tanh MLP -> scalar V(s)."""

    def __init__(self, obs_dim: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(obs))
        h = torch.tanh(self.fc2(h))
        return self.v(h).squeeze(-1)


# -----------------------------------------------------------------------------
# Rollout storage
# -----------------------------------------------------------------------------


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, 2), dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None
        self.size = size
        self.ptr = 0

    def reset(self):
        self.ptr = 0
        self.advantages = None
        self.returns = None

    def add(self, obs, action, log_prob, reward, value, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.ptr += 1

    def compute_advantages(self, last_value: float, gamma: float, lam: float):
        advantages = np.zeros(self.size, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(self.size)):
            next_value = last_value if t == self.size - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae
        returns = advantages + self.values
        return advantages, returns


def _compute_gae_arrays(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
    return advantages, advantages + values


def _apply_stage_config(stage: StageConfig) -> None:
    config.ENV_FORWARD_SPEED_REWARD_SCALE = stage.forward_speed_reward_scale
    config.ENV_COLLISION_PENALTY = stage.collision_penalty
    config.ENV_LAP_TIME_BONUS_SCALE = stage.lap_time_bonus_scale
    config.ENV_CENTERLINE_PENALTY_SCALE = stage.centerline_penalty_scale


def _new_rollout_episode_tracker(track_name: str) -> dict:
    return {
        "return": 0.0,
        "length": 0,
        "max_progress": 0.0,
        "max_forward_speed": 0.0,
        "forward_speed_sum": 0.0,
        "steer_abs_sum": 0.0,
        "steer_jerk_sum": 0.0,
        "lateral_abs_sum": 0.0,
        "sensor_min_sum": 0.0,
        "drifting_count": 0,
        "collisions": 0,
        "laps": 0,
        "last_steer": 0.0,
        "track": track_name,
    }


def _accumulate_rollout_episode(
    tracker: dict,
    reward: float,
    observation,
    forward_progress: float,
    action_np: np.ndarray,
    collision_delta: int,
    lap_completed: bool,
) -> None:
    tracker["return"] += reward
    tracker["length"] += 1
    if forward_progress > tracker["max_progress"]:
        tracker["max_progress"] = forward_progress
    if observation.forward_speed > tracker["max_forward_speed"]:
        tracker["max_forward_speed"] = observation.forward_speed
    tracker["forward_speed_sum"] += observation.forward_speed
    steer = float(action_np[0])
    tracker["steer_abs_sum"] += abs(steer)
    tracker["steer_jerk_sum"] += abs(steer - tracker["last_steer"])
    tracker["last_steer"] = steer
    tracker["lateral_abs_sum"] += abs(observation.lateral_speed)
    tracker["sensor_min_sum"] += observation.min_sensor_normalized
    if observation.is_drifting:
        tracker["drifting_count"] += 1
    tracker["collisions"] += max(0, int(collision_delta))
    if lap_completed:
        tracker["laps"] += 1


def _finalize_rollout_episode(tracker: dict) -> dict:
    n = max(1, tracker["length"])
    return {
        "return": tracker["return"],
        "length": tracker["length"],
        "max_progress": tracker["max_progress"],
        "max_forward_speed": tracker["max_forward_speed"],
        "mean_forward_speed": tracker["forward_speed_sum"] / n,
        "mean_abs_steer": tracker["steer_abs_sum"] / n,
        "mean_steer_jerk": tracker["steer_jerk_sum"] / n,
        "mean_abs_lateral": tracker["lateral_abs_sum"] / n,
        "mean_min_sensor": tracker["sensor_min_sum"] / n,
        "drift_fraction": tracker["drifting_count"] / n,
        "collisions": tracker["collisions"],
        "laps": tracker["laps"],
        "track": tracker["track"],
    }


def _rollout_worker(args: dict) -> dict:
    torch.set_num_threads(1)
    stage: StageConfig = args["stage"]
    cfg: RLConfig = args["cfg"]
    worker_id: int = args["worker_id"]
    seed: int = args["seed"]
    steps: int = args["steps"]
    obs_dim: int = args["obs_dim"]

    _apply_stage_config(stage)
    rng = random.Random(seed)
    track_name = rng.choice(stage.track_names)
    env = RaceEnv(
        max_steps=cfg.max_steps_per_episode,
        track_name=track_name,
        fast_observations=True,
    )
    obs, _ = env.reset(randomize_start=stage.randomize_start, rng=rng)

    actor = Actor(obs_dim, cfg.hidden_size, cfg.initial_log_std)
    critic = Critic(obs_dim, cfg.hidden_size)
    actor.load_state_dict(args["actor_state"])
    critic.load_state_dict(args["critic_state"])
    actor.eval()
    critic.eval()

    obs_buf = np.zeros((steps, obs_dim), dtype=np.float32)
    actions = np.zeros((steps, 2), dtype=np.float32)
    log_probs = np.zeros(steps, dtype=np.float32)
    rewards = np.zeros(steps, dtype=np.float32)
    values = np.zeros(steps, dtype=np.float32)
    dones = np.zeros(steps, dtype=np.float32)
    episodes: list[dict] = []
    tracker = _new_rollout_episode_tracker(env.simulation.track_name)

    for i in range(steps):
        obs_np = np.asarray(obs, dtype=np.float32)
        obs_t = torch.from_numpy(obs_np).unsqueeze(0)
        with torch.no_grad():
            dist = actor.dist(obs_t)
            raw_action = dist.sample()
            log_prob = dist.log_prob(raw_action).sum(-1)
            value = critic(obs_t)

        action_np = raw_action.squeeze(0).numpy()
        action_clipped = np.clip(action_np, -1.0, 1.0)
        total_reward = 0.0
        done = False

        for _ in range(cfg.frame_skip):
            (
                obs,
                reward,
                terminated,
                truncated,
                _progress_delta,
                collision_delta,
                lap_completed,
            ) = env.step_training(action_clipped)
            total_reward += reward
            _accumulate_rollout_episode(
                tracker,
                reward,
                env.simulation.observation,
                env.max_watermark,
                action_clipped,
                collision_delta,
                lap_completed,
            )
            if terminated or truncated:
                done = True
                break

        obs_buf[i] = obs_np
        actions[i] = action_np
        log_probs[i] = log_prob.item()
        rewards[i] = total_reward
        values[i] = value.item()
        dones[i] = 1.0 if done else 0.0

        if done:
            episodes.append(_finalize_rollout_episode(tracker))
            next_track = rng.choice(stage.track_names)
            if next_track != env.simulation.track_name:
                obs, _ = env.load_track(
                    next_track,
                    randomize_start=stage.randomize_start,
                    rng=rng,
                )
            else:
                obs, _ = env.reset(randomize_start=stage.randomize_start, rng=rng)
            tracker = _new_rollout_episode_tracker(env.simulation.track_name)

    obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        last_value = critic(obs_t).item()
    advantages, returns = _compute_gae_arrays(
        rewards,
        values,
        dones,
        last_value,
        cfg.gamma,
        cfg.gae_lambda,
    )
    return {
        "worker_id": worker_id,
        "obs": obs_buf,
        "actions": actions,
        "log_probs": log_probs,
        "rewards": rewards,
        "values": values,
        "dones": dones,
        "advantages": advantages,
        "returns": returns,
        "episodes": episodes,
        "last_obs": obs,
    }


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------


class PPOTrainer:
    def __init__(self, cfg: RLConfig):
        if not cfg.stages:
            raise ValueError("RLConfig.stages must contain at least one stage.")
        self.cfg = cfg

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.env = RaceEnv(
            max_steps=cfg.max_steps_per_episode,
            track_name=DEFAULT_TRACK_NAME,
            fast_observations=True,
        )
        probe_obs, _ = self.env.reset()
        self.obs_dim = len(probe_obs)

        self.actor = Actor(self.obs_dim, cfg.hidden_size, cfg.initial_log_std)
        self.critic = Critic(self.obs_dim, cfg.hidden_size)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.learning_rate,
        )

        self.rollout_workers = max(1, int(cfg.rollout_workers))
        self.effective_rollout_steps = cfg.rollout_steps * self.rollout_workers
        self.buffer = RolloutBuffer(self.effective_rollout_steps, self.obs_dim)

        self.total_steps = 0
        self.critic_pretrain_steps = 0
        self.updates = 0
        self.current_stage_idx = 0
        self.history: list[dict] = []
        self.recent_episodes: list[dict] = []  # rolling window across updates
        self.rollout_episodes: list[dict] = []  # episodes completed this rollout
        self.best_mean_return = -float("inf")
        self.rng = random.Random(cfg.seed)
        self.start_time = time.perf_counter()

        self._apply_stage(self._current_stage(), announce=True, prior_summary=None)
        self._reset_episode_tracker()
        self.current_obs, _ = self._reset_env()

    # ---- stage handling ----

    def _current_stage(self) -> StageConfig:
        return self.cfg.stages[self.current_stage_idx]

    def _stage_start_step(self) -> int:
        return sum(s.steps for s in self.cfg.stages[: self.current_stage_idx])

    def _apply_stage(self, stage: StageConfig, *, announce: bool, prior_summary: Optional[dict]):
        _apply_stage_config(stage)
        if not announce:
            return
        bar = "=" * 80
        print("")
        print(bar)
        print(f" STAGE {self.current_stage_idx + 1}/{len(self.cfg.stages)}  :  {stage.name}")
        print(bar)
        if prior_summary is not None:
            print(
                " prior stage summary: "
                f"episodes={prior_summary['episodes']:>4d}  "
                f"mean_return={prior_summary['mean_return']:+8.2f}  "
                f"mean_progress={prior_summary['mean_progress']:.3f}  "
                f"lap_pct={prior_summary['lap_pct']:5.1f}%  "
                f"collisions/ep={prior_summary['collisions_per_ep']:.2f}"
            )
        tracks_str = ", ".join(stage.track_names)
        if len(tracks_str) > 68:
            tracks_str = tracks_str[:65] + "..."
        print(f" tracks                  : {tracks_str}")
        print(f" randomize_start         : {stage.randomize_start}")
        print(f" forward_speed_reward    : {stage.forward_speed_reward_scale}")
        print(f" backward_speed_penalty  : {config.ENV_BACKWARD_SPEED_PENALTY_SCALE}")
        print(f" wrong_way_penalty       : {config.ENV_WRONG_WAY_PROGRESS_PENALTY_SCALE}")
        print(f" collision_penalty       : {stage.collision_penalty}")
        print(f" lap_time_bonus_scale    : {stage.lap_time_bonus_scale}")
        print(f" centerline_penalty_scale: {stage.centerline_penalty_scale}")
        print(f" entropy_coef            : {stage.entropy_coef}")
        print(f" stage length (steps)    : {stage.steps:,}")
        print(bar, flush=True)
        print("")

    def _advance_stage_if_needed(self):
        stage_end = self._stage_start_step() + self._current_stage().steps
        if (
            self.total_steps >= stage_end
            and self.current_stage_idx < len(self.cfg.stages) - 1
        ):
            prior_summary = self._summarize_recent_episodes()
            self.current_stage_idx += 1
            self._apply_stage(self._current_stage(), announce=True, prior_summary=prior_summary)

    # ---- env handling ----

    def _pick_track(self) -> str:
        names = self._current_stage().track_names
        return self.rng.choice(names) if len(names) > 1 else names[0]

    def _reset_env(self):
        stage = self._current_stage()
        track_name = self._pick_track()
        if track_name != self.env.simulation.track_name:
            return self.env.load_track(
                track_name, randomize_start=stage.randomize_start, rng=self.rng
            )
        return self.env.reset(randomize_start=stage.randomize_start, rng=self.rng)

    # ---- episode stats ----

    def _reset_episode_tracker(self):
        self.ep_return = 0.0
        self.ep_len = 0
        self.ep_max_progress = 0.0
        self.ep_max_forward_speed = 0.0
        self.ep_forward_speed_sum = 0.0
        self.ep_steer_abs_sum = 0.0
        self.ep_steer_jerk_sum = 0.0
        self.ep_lateral_abs_sum = 0.0
        self.ep_sensor_min_sum = 0.0
        self.ep_drifting_count = 0
        self.ep_collisions = 0
        self.ep_laps = 0
        self.ep_last_steer = 0.0
        self.ep_track_name = self.env.simulation.track_name

    def _accumulate_step(self, reward: float, info: dict, action_np: np.ndarray):
        self.ep_return += reward
        self.ep_len += 1

        fp = info.get("forward_progress", 0.0)
        if fp > self.ep_max_progress:
            self.ep_max_progress = fp

        fs = info.get("forward_speed", 0.0)
        self.ep_forward_speed_sum += fs
        if fs > self.ep_max_forward_speed:
            self.ep_max_forward_speed = fs

        steer = float(action_np[0])
        self.ep_steer_abs_sum += abs(steer)
        self.ep_steer_jerk_sum += abs(steer - self.ep_last_steer)
        self.ep_last_steer = steer

        sim_obs = self.env.simulation.observation
        self.ep_lateral_abs_sum += abs(sim_obs.lateral_speed)
        if sim_obs.vector is not None:
            self.ep_sensor_min_sum += sim_obs.min_sensor_normalized
        elif sim_obs.sensor_readings:
            self.ep_sensor_min_sum += min(
                r.normalized_distance for r in sim_obs.sensor_readings
            )
        if sim_obs.is_drifting:
            self.ep_drifting_count += 1

        self.ep_collisions += max(0, int(info.get("collision_delta", 0)))
        if info.get("lap_completed"):
            self.ep_laps += 1

    def _finalize_episode(self):
        n = max(1, self.ep_len)
        stats = {
            "return": self.ep_return,
            "length": self.ep_len,
            "max_progress": self.ep_max_progress,
            "max_forward_speed": self.ep_max_forward_speed,
            "mean_forward_speed": self.ep_forward_speed_sum / n,
            "mean_abs_steer": self.ep_steer_abs_sum / n,
            "mean_steer_jerk": self.ep_steer_jerk_sum / n,
            "mean_abs_lateral": self.ep_lateral_abs_sum / n,
            "mean_min_sensor": self.ep_sensor_min_sum / n,
            "drift_fraction": self.ep_drifting_count / n,
            "collisions": self.ep_collisions,
            "laps": self.ep_laps,
            "track": self.ep_track_name,
        }
        self.rollout_episodes.append(stats)
        self.recent_episodes.append(stats)
        window = self.cfg.recent_episode_window
        if len(self.recent_episodes) > window:
            del self.recent_episodes[: len(self.recent_episodes) - window]

    def _summarize_recent_episodes(self) -> dict:
        eps = self.recent_episodes
        if not eps:
            return {
                "episodes": 0,
                "mean_return": 0.0,
                "mean_progress": 0.0,
                "lap_pct": 0.0,
                "collisions_per_ep": 0.0,
            }
        mean_return = float(np.mean([e["return"] for e in eps]))
        mean_progress = float(np.mean([e["max_progress"] for e in eps]))
        lap_pct = 100.0 * sum(1 for e in eps if e["laps"] > 0) / len(eps)
        coll = float(np.mean([e["collisions"] for e in eps]))
        return {
            "episodes": len(eps),
            "mean_return": mean_return,
            "mean_progress": mean_progress,
            "lap_pct": lap_pct,
            "collisions_per_ep": coll,
        }

    # ---- rollout ----

    def collect_rollout(self, *, count_training_steps: bool = True, pool=None) -> float:
        if self.rollout_workers > 1:
            if pool is None:
                raise RuntimeError(
                    "collect_rollout requires a rollout pool when rollout_workers > 1"
                )
            return self._collect_rollout_parallel(
                count_training_steps=count_training_steps,
                pool=pool,
            )

        self.actor.eval()
        self.critic.eval()
        self.buffer.reset()
        self.rollout_episodes = []

        obs = self.current_obs
        for _ in range(self.cfg.rollout_steps):
            obs_np = np.asarray(obs, dtype=np.float32)
            obs_t = torch.from_numpy(obs_np).unsqueeze(0)
            with torch.no_grad():
                dist = self.actor.dist(obs_t)
                raw_action = dist.sample()
                log_prob = dist.log_prob(raw_action).sum(-1)
                value = self.critic(obs_t)

            action_np = raw_action.squeeze(0).numpy()
            action_clipped = np.clip(action_np, -1.0, 1.0)

            # frame-skip: repeat the chosen action for N sim steps, sum rewards
            total_reward = 0.0
            done = False
            step_result = None
            for _ in range(self.cfg.frame_skip):
                (
                    step_obs,
                    reward,
                    terminated,
                    truncated,
                    progress_delta,
                    collision_delta,
                    lap_completed,
                ) = self.env.step_training(action_clipped)
                total_reward += reward
                info = {
                    "forward_progress": self.env.max_watermark,
                    "forward_speed": self.env.simulation.observation.forward_speed,
                    "collision_delta": collision_delta,
                    "lap_completed": lap_completed,
                }
                self._accumulate_step(reward, info, action_clipped)
                if terminated or truncated:
                    done = True
                    break

            self.buffer.add(
                obs=obs_np,
                action=action_np,
                log_prob=log_prob.item(),
                reward=total_reward,
                value=value.item(),
                done=1.0 if done else 0.0,
            )
            if count_training_steps:
                self.total_steps += 1
            else:
                self.critic_pretrain_steps += 1

            if done:
                self._finalize_episode()
                self._reset_episode_tracker()
                obs, _ = self._reset_env()
            else:
                obs = step_obs

        self.current_obs = obs
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            last_value = self.critic(obs_t).item()
        return last_value

    def _collect_rollout_parallel(self, *, count_training_steps: bool, pool) -> float:
        self.actor.eval()
        self.critic.eval()
        self.buffer.reset()
        self.rollout_episodes = []

        actor_state = {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}
        critic_state = {k: v.detach().cpu() for k, v in self.critic.state_dict().items()}
        base_seed = self.cfg.seed + self.updates * 100_000 + self.current_stage_idx * 10_000
        jobs = [
            {
                "worker_id": worker_id,
                "seed": base_seed + worker_id,
                "steps": self.cfg.rollout_steps,
                "obs_dim": self.obs_dim,
                "cfg": self.cfg,
                "stage": self._current_stage(),
                "actor_state": actor_state,
                "critic_state": critic_state,
            }
            for worker_id in range(self.rollout_workers)
        ]
        results = pool.map(_rollout_worker, jobs)
        results.sort(key=lambda item: item["worker_id"])

        offset = 0
        for result in results:
            n = len(result["rewards"])
            self.buffer.obs[offset : offset + n] = result["obs"]
            self.buffer.actions[offset : offset + n] = result["actions"]
            self.buffer.log_probs[offset : offset + n] = result["log_probs"]
            self.buffer.rewards[offset : offset + n] = result["rewards"]
            self.buffer.values[offset : offset + n] = result["values"]
            self.buffer.dones[offset : offset + n] = result["dones"]
            if self.buffer.advantages is None:
                self.buffer.advantages = np.zeros(self.buffer.size, dtype=np.float32)
                self.buffer.returns = np.zeros(self.buffer.size, dtype=np.float32)
            self.buffer.advantages[offset : offset + n] = result["advantages"]
            self.buffer.returns[offset : offset + n] = result["returns"]
            self.rollout_episodes.extend(result["episodes"])
            offset += n

        if count_training_steps:
            self.total_steps += self.buffer.size
        else:
            self.critic_pretrain_steps += self.buffer.size

        self.recent_episodes.extend(self.rollout_episodes)
        window = self.cfg.recent_episode_window
        if len(self.recent_episodes) > window:
            del self.recent_episodes[: len(self.recent_episodes) - window]

        if results:
            self.current_obs = results[0]["last_obs"]
        return 0.0

    # ---- PPO update ----

    def update(self, last_value: float) -> dict:
        cfg = self.cfg
        if self.buffer.advantages is None or self.buffer.returns is None:
            advantages, returns = self.buffer.compute_advantages(
                last_value, cfg.gamma, cfg.gae_lambda
            )
        else:
            advantages = self.buffer.advantages
            returns = self.buffer.returns

        obs_t = torch.from_numpy(self.buffer.obs)
        actions_t = torch.from_numpy(self.buffer.actions)
        old_log_probs_t = torch.from_numpy(self.buffer.log_probs)
        old_values_t = torch.from_numpy(self.buffer.values)
        advantages_t = torch.from_numpy(advantages)
        returns_t = torch.from_numpy(returns)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = self.buffer.size
        indices = np.arange(n)

        actor_losses, critic_losses, entropies, kls, clip_fracs = [], [], [], [], []
        entropy_coef = self._current_stage().entropy_coef

        self.actor.train()
        self.critic.train()

        for _ in range(cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, cfg.minibatch_size):
                batch_idx = indices[start : start + cfg.minibatch_size]
                b_obs = obs_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_lp = old_log_probs_t[batch_idx]
                b_old_v = old_values_t[batch_idx]
                b_adv = advantages_t[batch_idx]
                b_ret = returns_t[batch_idx]

                dist = self.actor.dist(b_obs)
                new_log_probs = dist.log_prob(b_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(b_obs)
                if cfg.value_clip_range is None:
                    critic_loss = 0.5 * (values - b_ret).pow(2).mean()
                else:
                    values_clipped = b_old_v + torch.clamp(
                        values - b_old_v,
                        -cfg.value_clip_range,
                        cfg.value_clip_range,
                    )
                    vf_loss1 = (values - b_ret).pow(2)
                    vf_loss2 = (values_clipped - b_ret).pow(2)
                    critic_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                loss = actor_loss + cfg.value_coef * critic_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (b_old_lp - new_log_probs).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_range).float().mean().item()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.item())
                kls.append(approx_kl)
                clip_fracs.append(clip_frac)

        y_pred = self.buffer.values
        y_true = returns
        var_y = float(np.var(y_true))
        explained_var = 1.0 - float(np.var(y_true - y_pred)) / (var_y + 1e-8)

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(kls)),
            "clip_frac": float(np.mean(clip_fracs)),
            "explained_var": float(explained_var),
            "log_std": self.actor.log_std.detach().cpu().tolist(),
        }

    # ---- critic prelearning ----

    def pretrain_critic(self, updates: Optional[int] = None) -> None:
        """Train only the critic from rollouts generated by the current actor.

        This is useful after an actor-only warm-start: the policy is already
        meaningful, but a random critic gives bad value baselines for the first
        PPO updates. The actor is kept frozen while the critic learns returns
        produced by that actor.
        """
        cfg = self.cfg
        updates = cfg.critic_pretrain_updates if updates is None else updates
        if updates <= 0:
            print("[critic-prelearn] skipped (updates <= 0)", flush=True)
            return

        optimizer = optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        bar = "=" * 80
        print("")
        print(bar)
        print(" CRITIC PRELEARNING")
        print(bar)
        print(f" updates                : {updates:,}")
        print(f" rollout_steps / worker : {cfg.rollout_steps:,}")
        print(f" rollout_workers        : {self.rollout_workers}")
        print(f" effective batch        : {self.buffer.size:,}")
        print(f" epochs / batch         : {cfg.critic_pretrain_epochs} / {cfg.minibatch_size}")
        print(f" actor                  : frozen, sampled for rollout actions")
        print(f" critic                 : train from actor-generated returns")
        print(f" stage                  : {self.current_stage_idx + 1}/{len(cfg.stages)} [{self._current_stage().name}]")
        print(bar, flush=True)
        print("")

        pool = self._make_rollout_pool()
        try:
            for update_idx in range(1, updates + 1):
                last_value = self.collect_rollout(count_training_steps=False, pool=pool)
                if self.buffer.returns is None:
                    _advantages, returns = self.buffer.compute_advantages(
                        last_value, cfg.gamma, cfg.gae_lambda
                    )
                else:
                    returns = self.buffer.returns

                obs_t = torch.from_numpy(self.buffer.obs)
                returns_t = torch.from_numpy(returns)
                indices = np.arange(self.buffer.size)
                losses: list[float] = []

                self.actor.eval()
                self.critic.train()
                for _ in range(cfg.critic_pretrain_epochs):
                    np.random.shuffle(indices)
                    for start in range(0, self.buffer.size, cfg.minibatch_size):
                        batch_idx = indices[start : start + cfg.minibatch_size]
                        values = self.critic(obs_t[batch_idx])
                        loss = 0.5 * (values - returns_t[batch_idx]).pow(2).mean()

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                        optimizer.step()
                        losses.append(loss.item())

                with torch.no_grad():
                    predicted = self.critic(obs_t).detach().cpu().numpy()
                var_y = float(np.var(returns))
                explained_var = 1.0 - float(np.var(returns - predicted)) / (var_y + 1e-8)
                diag = {
                    "update": update_idx,
                    "updates": updates,
                    "critic_loss": float(np.mean(losses)),
                    "return_mean": float(np.mean(returns)),
                    "return_std": float(np.std(returns)),
                    "explained_var": float(explained_var),
                }
                self._print_critic_pretrain_stats(diag)
        finally:
            self._close_rollout_pool(pool)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.learning_rate,
        )
        self.recent_episodes = []
        self.rollout_episodes = []
        self._reset_episode_tracker()
        print(
            "[critic-prelearn] complete; PPO optimizer reset and PPO episode window cleared",
            flush=True,
        )
        print("")

    # ---- main loop ----

    def _make_rollout_pool(self):
        if self.rollout_workers <= 1:
            return None
        ctx = mp.get_context("spawn")
        return ctx.Pool(processes=self.rollout_workers)

    def _close_rollout_pool(self, pool) -> None:
        if pool is None:
            return
        pool.close()
        pool.join()

    def train(self):
        cfg = self.cfg
        total_updates_target = cfg.total_steps // self.buffer.size
        bar = "=" * 80
        print(bar)
        print(" PPO training")
        print(bar)
        print(f" total_steps            : {cfg.total_steps:,}")
        print(f" rollout_steps / worker : {cfg.rollout_steps:,}")
        print(f" rollout_workers        : {self.rollout_workers}")
        print(f" effective batch/update : {self.buffer.size:,}")
        print(f" frame_skip             : {cfg.frame_skip}")
        print(f" updates (target)       : {total_updates_target:,}")
        print(f" obs_dim                : {self.obs_dim}")
        print(f" hidden_size            : {cfg.hidden_size}")
        print(f" gamma / gae_lambda     : {cfg.gamma} / {cfg.gae_lambda}")
        print(f" clip_range             : {cfg.clip_range}")
        print(f" value_clip_range       : {cfg.value_clip_range}")
        print(f" lr / n_epochs / batch  : {cfg.learning_rate} / {cfg.n_epochs} / {cfg.minibatch_size}")
        print(f" critic prelearn        : {cfg.critic_pretrain_updates} updates x {cfg.critic_pretrain_epochs} epochs")
        print(f" seed                   : {cfg.seed}")
        print(f" checkpoint             : {cfg.checkpoint_path}")
        print(f" policy JSON (best)     : {cfg.policy_json_path}")
        print(f" policy JSON (latest)   : {_latest_json_path(cfg.policy_json_path)}")
        print(f" PPO metrics log        : {cfg.ppo_log_path}")
        print(bar, flush=True)
        print("")

        pool = self._make_rollout_pool()
        try:
            while self.total_steps < cfg.total_steps:
                self._advance_stage_if_needed()
                last_value = self.collect_rollout(pool=pool)
                diag = self.update(last_value)
                self.updates += 1

                # After every update: refresh the latest-weights JSON snapshot
                # and persist the full .pt checkpoint (so resume is always current).
                self.export_policy_json(latest=True)
                self.save_checkpoint()

                # On best-return improvement: also write the best-weights JSON.
                improvement_info = self._maybe_save_best()
                self._write_ppo_update_log(diag, improvement_info)

                if self.updates % cfg.print_every_updates == 0:
                    self._print_stats(diag, improvement_info)
        finally:
            self._close_rollout_pool(pool)

        # Safety net: persist final .pt so a training run always has a resume point,
        # even if the final updates did not improve the best mean return.
        self.save_checkpoint()
        self.export_policy_json(latest=True)

        print("")
        print(bar)
        print(" training complete")
        print(f"  checkpoint       : {cfg.checkpoint_path}")
        print(f"  best policy JSON : {cfg.policy_json_path}")
        print(f"  latest policy    : {_latest_json_path(cfg.policy_json_path)}")
        print(bar, flush=True)

    # ---- save cadence ----

    def _maybe_save_best(self) -> Optional[dict]:
        """If the current rollout's mean return beats the best seen so far,
        write the best-weights JSON. Returns improvement info for logging, or None.
        """
        if not self.rollout_episodes:
            return None
        mean_return = float(np.mean([e["return"] for e in self.rollout_episodes]))
        if mean_return <= self.best_mean_return:
            return None
        prev = self.best_mean_return
        self.best_mean_return = mean_return
        self.export_policy_json(latest=False)
        return {"prev": prev, "new": mean_return}

    # ---- logging ----

    def _print_stats(self, diag: dict, improvement_info: Optional[dict] = None):
        bar = "=" * 80
        stage = self._current_stage()
        stage_start = self._stage_start_step()
        stage_pos = self.total_steps - stage_start
        stage_pct = 100.0 * stage_pos / max(1, stage.steps)

        elapsed = time.perf_counter() - self.start_time
        steps_per_sec = self.total_steps / max(elapsed, 1e-6)
        hh = int(elapsed // 3600)
        mm = int((elapsed % 3600) // 60)
        ss = int(elapsed % 60)
        elapsed_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

        rollout_eps = self.rollout_episodes

        print(bar)
        print(
            f" update {self.updates:>5d}   step {self.total_steps:>10,}/{self.cfg.total_steps:,}   "
            f"stage {self.current_stage_idx + 1}/{len(self.cfg.stages)} [{stage.name}] "
            f"{stage_pos:,}/{stage.steps:,} ({stage_pct:5.1f}%)"
        )
        print(
            f" elapsed {elapsed_str}   rate {steps_per_sec:7.1f} steps/s   "
            f"episodes_this_rollout {len(rollout_eps):>3d}   recent_window {len(self.recent_episodes):>3d}"
        )
        print("-" * 80)

        if rollout_eps:
            self._print_behavior_block("this rollout", rollout_eps)
        else:
            print(" this rollout       : (no episodes completed — episodes still ongoing)")

        if self.recent_episodes and len(self.recent_episodes) > len(rollout_eps):
            print("-" * 80)
            self._print_behavior_block(
                f"recent (last {len(self.recent_episodes)} eps)", self.recent_episodes
            )

        print("-" * 80)
        sigma = [math.exp(x) for x in diag["log_std"]]
        print(
            f" ppo   actor_loss {diag['actor_loss']:+.4f}   critic_loss {diag['critic_loss']:8.3f}   "
            f"entropy {diag['entropy']:+.3f}"
        )
        print(
            f"       approx_kl  {diag['approx_kl']:+.4f}   clip_frac   {diag['clip_frac']:8.3f}   "
            f"exp_var {diag['explained_var']:+.3f}"
        )
        print(
            f"       sigma(steer) {sigma[0]:.3f}   sigma(accel) {sigma[1]:.3f}   "
            f"lr {self.cfg.learning_rate:.1e}   entropy_coef {stage.entropy_coef}"
        )
        print(bar, flush=True)
        print("")

        self.history.append(
            {
                "update": self.updates,
                "step": self.total_steps,
                "stage": self.current_stage_idx,
                "stage_name": stage.name,
                "rollout_episodes": len(rollout_eps),
                "recent_mean_return": float(
                    np.mean([e["return"] for e in self.recent_episodes])
                )
                if self.recent_episodes
                else None,
                "actor_loss": diag["actor_loss"],
                "critic_loss": diag["critic_loss"],
                "entropy": diag["entropy"],
                "approx_kl": diag["approx_kl"],
                "clip_frac": diag["clip_frac"],
                "explained_var": diag["explained_var"],
            }
        )

        if improvement_info is not None:
            prev = improvement_info["prev"]
            new = improvement_info["new"]
            prev_str = "-inf" if not math.isfinite(prev) else f"{prev:+.2f}"
            print(
                f" [best improved {prev_str} -> {new:+.2f}  (ppo_policy.json updated)]",
                flush=True,
            )
            print("")

    def _print_critic_pretrain_stats(self, diag: dict):
        bar = "-" * 80
        rollout_eps = self.rollout_episodes
        print(bar)
        print(
            f" critic-prelearn update {diag['update']:>4d}/{diag['updates']:<4d} "
            f"pretrain_steps {self.critic_pretrain_steps:>10,}"
        )
        print(
            f" value_fit loss {diag['critic_loss']:9.3f}   "
            f"return_mean {diag['return_mean']:+9.2f}   "
            f"return_std {diag['return_std']:8.2f}   "
            f"exp_var {diag['explained_var']:+.3f}"
        )
        if rollout_eps:
            self._print_behavior_block("critic-prelearn rollout", rollout_eps)
        else:
            print(" critic-prelearn rollout: (no episodes completed - episodes still ongoing)")
        print(bar, flush=True)
        print("")

        self.history.append(
            {
                "phase": "critic_prelearn",
                "critic_pretrain_update": diag["update"],
                "critic_pretrain_steps": self.critic_pretrain_steps,
                "stage": self.current_stage_idx,
                "stage_name": self._current_stage().name,
                "rollout_episodes": len(rollout_eps),
                "critic_loss": diag["critic_loss"],
                "return_mean": diag["return_mean"],
                "return_std": diag["return_std"],
                "explained_var": diag["explained_var"],
            }
        )
        self._write_critic_pretrain_log(diag)

    def _write_ppo_update_log(self, diag: dict, improvement_info: Optional[dict]):
        stage = self._current_stage()
        sigma = [math.exp(x) for x in diag["log_std"]]
        best_improvement = None
        if improvement_info is not None:
            prev = improvement_info["prev"]
            best_improvement = {
                "prev": None if not math.isfinite(prev) else prev,
                "new": improvement_info["new"],
            }
        record = {
            "phase": "ppo_update",
            "update": self.updates,
            "step": self.total_steps,
            "stage_index": self.current_stage_idx,
            "stage_name": stage.name,
            "stage_step": self.total_steps - self._stage_start_step(),
            "stage_total_steps": stage.steps,
            "stage": self._stage_config_dict(stage),
            "ppo": {
                "actor_loss": diag["actor_loss"],
                "critic_loss": diag["critic_loss"],
                "entropy": diag["entropy"],
                "approx_kl": diag["approx_kl"],
                "clip_frac": diag["clip_frac"],
                "explained_var": diag["explained_var"],
                "log_std_steer": diag["log_std"][0],
                "log_std_accel": diag["log_std"][1],
                "sigma_steer": sigma[0],
                "sigma_accel": sigma[1],
            },
            "hyperparams": self._hyperparam_log_dict(),
            "rollout": self._episode_summary(self.rollout_episodes),
            "recent": self._episode_summary(self.recent_episodes),
            "per_track_rollout": self._per_track_summary(self.rollout_episodes),
            "best_improvement": best_improvement,
        }
        self._append_training_log(record)

    def _write_critic_pretrain_log(self, diag: dict):
        stage = self._current_stage()
        record = {
            "phase": "critic_prelearn",
            "critic_pretrain_update": diag["update"],
            "critic_pretrain_updates": diag["updates"],
            "critic_pretrain_steps": self.critic_pretrain_steps,
            "stage_index": self.current_stage_idx,
            "stage_name": stage.name,
            "stage": self._stage_config_dict(stage),
            "critic": {
                "loss": diag["critic_loss"],
                "return_mean": diag["return_mean"],
                "return_std": diag["return_std"],
                "explained_var": diag["explained_var"],
            },
            "hyperparams": self._hyperparam_log_dict(),
            "rollout": self._episode_summary(self.rollout_episodes),
            "per_track_rollout": self._per_track_summary(self.rollout_episodes),
        }
        self._append_training_log(record)

    def _append_training_log(self, record: dict):
        path = Path(self.cfg.ppo_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")

    def _hyperparam_log_dict(self):
        return {
            "learning_rate": self.cfg.learning_rate,
            "rollout_steps": self.cfg.rollout_steps,
            "rollout_workers": self.rollout_workers,
            "effective_rollout_steps": self.buffer.size,
            "frame_skip": self.cfg.frame_skip,
            "n_epochs": self.cfg.n_epochs,
            "minibatch_size": self.cfg.minibatch_size,
            "gamma": self.cfg.gamma,
            "gae_lambda": self.cfg.gae_lambda,
            "clip_range": self.cfg.clip_range,
            "value_clip_range": self.cfg.value_clip_range,
            "value_coef": self.cfg.value_coef,
            "max_grad_norm": self.cfg.max_grad_norm,
            "critic_pretrain_updates": self.cfg.critic_pretrain_updates,
            "critic_pretrain_epochs": self.cfg.critic_pretrain_epochs,
            "backward_speed_penalty_scale": config.ENV_BACKWARD_SPEED_PENALTY_SCALE,
            "wrong_way_progress_penalty_scale": config.ENV_WRONG_WAY_PROGRESS_PENALTY_SCALE,
        }

    def _stage_config_dict(self, stage: StageConfig):
        return {
            "name": stage.name,
            "steps": stage.steps,
            "track_count": len(stage.track_names),
            "track_names": list(stage.track_names),
            "randomize_start": stage.randomize_start,
            "forward_speed_reward_scale": stage.forward_speed_reward_scale,
            "backward_speed_penalty_scale": config.ENV_BACKWARD_SPEED_PENALTY_SCALE,
            "wrong_way_progress_penalty_scale": config.ENV_WRONG_WAY_PROGRESS_PENALTY_SCALE,
            "entropy_coef": stage.entropy_coef,
            "collision_penalty": stage.collision_penalty,
            "lap_time_bonus_scale": stage.lap_time_bonus_scale,
            "centerline_penalty_scale": stage.centerline_penalty_scale,
        }

    def _episode_summary(self, eps: list[dict]):
        if not eps:
            return {"episodes": 0}

        def values(key):
            return [e[key] for e in eps]

        def mean(key):
            return float(np.mean(values(key)))

        def minv(key):
            return float(min(values(key)))

        def maxv(key):
            return float(max(values(key)))

        def std(key):
            return float(np.std(values(key)))

        laps_total = int(sum(e["laps"] for e in eps))
        return {
            "episodes": len(eps),
            "return_mean": mean("return"),
            "return_min": minv("return"),
            "return_max": maxv("return"),
            "return_std": std("return"),
            "length_mean": mean("length"),
            "length_min": minv("length"),
            "length_max": maxv("length"),
            "progress_mean": mean("max_progress"),
            "progress_max": maxv("max_progress"),
            "collisions_mean": mean("collisions"),
            "collisions_max": maxv("collisions"),
            "laps_total": laps_total,
            "full_lap_pct": 100.0 * sum(1 for e in eps if e["laps"] > 0) / len(eps),
        }

    def _per_track_summary(self, eps: list[dict]):
        by_track: dict[str, list[dict]] = {}
        for e in eps:
            by_track.setdefault(e["track"], []).append(e)
        return {
            track_name: self._episode_summary(track_eps)
            for track_name, track_eps in sorted(by_track.items())
        }

    def _print_behavior_block(self, label: str, eps: list[dict]):
        def mean(key):
            return float(np.mean([e[key] for e in eps]))

        def maxv(key):
            return float(max(e[key] for e in eps))

        def minv(key):
            return float(min(e[key] for e in eps))

        def std(key):
            return float(np.std([e[key] for e in eps]))

        laps_total = sum(e["laps"] for e in eps)
        lap_completion_pct = 100.0 * sum(1 for e in eps if e["laps"] > 0) / len(eps)

        print(f" {label}")
        print(
            f"   return        mean {mean('return'):+9.2f}   min {minv('return'):+9.2f}   "
            f"max {maxv('return'):+9.2f}   std {std('return'):8.2f}"
        )
        print(
            f"   ep_length     mean {mean('length'):7.1f}   max {maxv('length'):5.0f}   min {minv('length'):5.0f}"
        )
        print(
            f"   progress      mean {mean('max_progress'):7.3f}   max {maxv('max_progress'):7.3f}   "
            f"laps_total {laps_total:3d}   full_lap_pct {lap_completion_pct:5.1f}%"
        )
        print(
            f"   speed (px/s)  mean_fwd {mean('mean_forward_speed'):7.1f}   "
            f"max_fwd {maxv('max_forward_speed'):7.1f}"
        )
        print(
            f"   steering      mean|s| {mean('mean_abs_steer'):5.3f}   "
            f"mean_jerk {mean('mean_steer_jerk'):5.3f}"
        )
        print(
            f"   slip          mean|lat| {mean('mean_abs_lateral'):6.1f} px/s   "
            f"drifting_time {100 * mean('drift_fraction'):5.1f}%"
        )
        print(
            f"   wall_prox     mean_min_sensor {mean('mean_min_sensor'):5.3f}   "
            f"collisions/ep {mean('collisions'):5.2f}"
        )
        # per-track breakdown if more than one track appears
        by_track: dict[str, list[dict]] = {}
        for e in eps:
            by_track.setdefault(e["track"], []).append(e)
        if len(by_track) > 1:
            print(f"   per-track:")
            for track_name, track_eps in sorted(by_track.items()):
                tp = float(np.mean([e["max_progress"] for e in track_eps]))
                tc = float(np.mean([e["collisions"] for e in track_eps]))
                tlp = 100.0 * sum(1 for e in track_eps if e["laps"] > 0) / len(track_eps)
                print(
                    f"     {track_name:<22s} n={len(track_eps):2d}   "
                    f"progress {tp:.3f}   lap_pct {tlp:5.1f}%   col/ep {tc:.2f}"
                )

    # ---- persistence ----

    def save_checkpoint(self):
        path = Path(self.cfg.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "critic_pretrain_steps": self.critic_pretrain_steps,
            "updates": self.updates,
            "current_stage_idx": self.current_stage_idx,
            "history": self.history,
            "recent_episodes": self.recent_episodes,
            "best_mean_return": self.best_mean_return,
            "rng_state": self.rng.getstate(),
            "np_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "obs_dim": self.obs_dim,
            "hidden_size": self.cfg.hidden_size,
            "rollout_workers": self.rollout_workers,
            "stages": [stage.__dict__ for stage in self.cfg.stages],
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("obs_dim") != self.obs_dim:
            raise ValueError(
                f"Checkpoint obs_dim={state.get('obs_dim')} does not match env obs_dim={self.obs_dim}."
            )
        if state.get("hidden_size") != self.cfg.hidden_size:
            raise ValueError(
                f"Checkpoint hidden_size={state.get('hidden_size')} does not match cfg {self.cfg.hidden_size}."
            )
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.total_steps = state["total_steps"]
        self.critic_pretrain_steps = state.get("critic_pretrain_steps", 0)
        self.updates = state["updates"]
        self.current_stage_idx = state["current_stage_idx"]
        self.history = state.get("history", [])
        self.recent_episodes = state.get("recent_episodes", [])
        self.best_mean_return = state.get("best_mean_return", -float("inf"))
        self.rng.setstate(state["rng_state"])
        np.random.set_state(state["np_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        self._apply_stage(self._current_stage(), announce=True, prior_summary=None)
        self._reset_episode_tracker()
        self.current_obs, _ = self._reset_env()

    def export_policy_json(self, *, latest: bool):
        data = self.actor.export_mlp_dict()
        data["metadata"] = self._policy_metadata(latest=latest)
        if latest:
            path = Path(_latest_json_path(self.cfg.policy_json_path))
        else:
            path = Path(self.cfg.policy_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def _policy_metadata(self, *, latest: bool) -> dict:
        stage = self._current_stage()
        best = self.best_mean_return
        return {
            "trained_steps": int(self.total_steps),
            "updates": int(self.updates),
            "stage_index": int(self.current_stage_idx),
            "stage_name": stage.name,
            "total_stages": len(self.cfg.stages),
            "best_mean_return": None if not math.isfinite(best) else float(best),
            "hidden_size": int(self.cfg.hidden_size),
            "obs_dim": int(self.obs_dim),
            "snapshot": "latest" if latest else "best",
        }

    # ---- stage jumping ----

    def jump_to_stage(self, idx: int) -> None:
        """Skip straight to curriculum stage ``idx`` (0-based), as if earlier
        stages had completed. Weights are preserved (call this *after* a
        warm-start if you want the prelearned actor/critic); the optimizer is
        reset so it doesn't carry over momentum from "imagined" prior
        training, and the rolling episode stats are cleared so the new
        stage's log isn't contaminated by an empty window.
        """
        if not (0 <= idx < len(self.cfg.stages)):
            raise ValueError(
                f"stage index {idx} out of range 0..{len(self.cfg.stages) - 1}"
            )
        stage_start_step = sum(s.steps for s in self.cfg.stages[:idx])
        self.current_stage_idx = idx
        self.total_steps = stage_start_step
        self.critic_pretrain_steps = 0
        self.updates = 0
        self.history = []
        self.recent_episodes = []
        self.rollout_episodes = []
        self.best_mean_return = -float("inf")
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.cfg.learning_rate,
        )
        self.start_time = time.perf_counter()
        self._apply_stage(self._current_stage(), announce=True, prior_summary=None)
        self._reset_episode_tracker()
        self.current_obs, _ = self._reset_env()

    # ---- warm start ----

    def warm_start_from_checkpoint(
        self,
        path: str,
        *,
        transfer_critic: bool = True,
        reset_log_std: Optional[tuple[float, float]] = None,
    ) -> None:
        """Load weights from an existing .pt checkpoint but
        reset every other piece of training state (optimizer, step counter,
        curriculum position, best return, history, RNG-derived caches). Use
        this to bootstrap a new curriculum with a previously learned network.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        prior_obs_dim = state.get("obs_dim")
        prior_hidden = state.get("hidden_size")
        if prior_obs_dim is not None and prior_obs_dim != self.obs_dim:
            raise ValueError(
                f"Warm-start checkpoint obs_dim={prior_obs_dim} does not match env obs_dim={self.obs_dim}."
            )
        if prior_hidden is not None and prior_hidden != self.cfg.hidden_size:
            raise ValueError(
                f"Warm-start checkpoint hidden_size={prior_hidden} does not match cfg {self.cfg.hidden_size}."
            )
        self.actor.load_state_dict(state["actor"])
        if reset_log_std is not None:
            with torch.no_grad():
                self.actor.log_std.copy_(
                    torch.tensor(reset_log_std, dtype=self.actor.log_std.dtype)
                )
        if transfer_critic:
            self.critic.load_state_dict(state["critic"])
        else:
            self.critic = Critic(self.obs_dim, self.cfg.hidden_size)
        # Fresh optimizer; keeping the prior optimizer state would carry over
        # momentum/learning-rate bookkeeping that no longer matches the new
        # curriculum's stage weights, defeating the purpose of warm-start.
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.cfg.learning_rate,
        )
        self.total_steps = 0
        self.critic_pretrain_steps = 0
        self.updates = 0
        self.current_stage_idx = 0
        self.history = []
        self.recent_episodes = []
        self.rollout_episodes = []
        self.best_mean_return = -float("inf")
        self.start_time = time.perf_counter()
        self._apply_stage(self._current_stage(), announce=True, prior_summary=None)
        self._reset_episode_tracker()
        self.current_obs, _ = self._reset_env()
        prior_steps = state.get("total_steps", "?")
        critic_msg = "actor+critic" if transfer_critic else "actor only"
        sigma = [math.exp(x) for x in self.actor.log_std.detach().cpu().tolist()]
        print(
            f"[warm-start] {critic_msg} loaded from {path}  "
            f"(prior trained_steps={prior_steps}, optimizer/curriculum reset, "
            f"sigma steer={sigma[0]:.3f}, accel={sigma[1]:.3f})",
            flush=True,
        )


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------


def make_default_stages(
    stage1_steps: int,
    stage2_steps: int,
    stage3_steps: int,
) -> list[StageConfig]:
    """Legacy 3-stage curriculum kept for back-compat. Prefer
    :func:`make_progressive_stages` for new runs.
    """
    return [
        StageConfig(
            name="warmup",
            steps=stage1_steps,
            track_names=[DEFAULT_TRACK_NAME],
            randomize_start=False,
            forward_speed_reward_scale=0.0,
            entropy_coef=0.01,
        ),
        StageConfig(
            name="generalize",
            steps=stage2_steps,
            track_names=available_track_names(),
            randomize_start=True,
            forward_speed_reward_scale=0.0,
            entropy_coef=0.005,
        ),
        StageConfig(
            name="race_fast",
            steps=stage3_steps,
            track_names=available_track_names(),
            randomize_start=True,
            forward_speed_reward_scale=0.0,
            entropy_coef=0.002,
        ),
    ]


def make_progressive_stages(
    *,
    stage_basics_steps: int,
    stage_basics_reverse_steps: int,
    stage_intermediate_steps: int,
    stage_full_steps: int,
    stage_race_fast_steps: int,
) -> list[StageConfig]:
    """Progressive curriculum: nail the basics first, widen the track set,
    then push for speed.

    Stage order and intent:

    1. ``basics_forward`` — only the simplest short tracks, fixed start,
       high entropy. The goal is "finish a lap slowly", not fast driving.
    2. ``basics_both_directions`` — same basics plus their reversed variants
       (so the policy sees both left-handed and right-handed corners).
       Random start begins here so the policy doesn't overfit to the spawn
       point. Still no speed reward.
    3. ``intermediate`` — basics + intermediate tracks (fwd + reversed).
       More variety; still no explicit speed shaping.
    4. ``full_grid`` — every registered track (basic + intermediate + the
       three original advanced tracks), both directions. Still no speed
       shaping: the policy should first learn to handle everything before
       we push it to go fast.
    5. ``race_fast`` — same track set as stage 4, but lower entropy so the
       policy tightens its lines without extra forward-speed shaping.
    """
    basics_fwd = list(basic_track_names(include_reverse=False))
    basics_all = list(basic_track_names(include_reverse=True))
    intermediate_all = list(intermediate_track_names(include_reverse=True))
    advanced_all = list(advanced_track_names(include_reverse=True))
    full_grid = basics_all + intermediate_all + advanced_all

    # Reward-shaping schedule. Design goals, per stage:
    #   collision_penalty     — start high so crashes always hurt more than
    #                           a fast lap pays; relax toward race_fast.
    #   lap_time_bonus_scale  — reduced from the original 400 so one crash
    #                           can't be papered over by a single fast lap
    #                           bonus.
    #   centerline_penalty_scale — small "stay near middle" nudge. Strongest
    #                           in basics to teach clean lines; decays to 0
    #                           in race_fast so the policy can pick late
    #                           apexes without being punished for it.
    return [
        StageConfig(
            name="basics_forward",
            steps=stage_basics_steps,
            track_names=basics_fwd,
            randomize_start=False,
            forward_speed_reward_scale=0.0,
            entropy_coef=0.015,
            collision_penalty=500.0,
            lap_time_bonus_scale=80.0,
            centerline_penalty_scale=0.003,
        ),
        StageConfig(
            name="basics_both_directions",
            steps=stage_basics_reverse_steps,
            track_names=basics_all,
            randomize_start=True,
            forward_speed_reward_scale=0.0,
            entropy_coef=0.01,
            collision_penalty=600.0,
            lap_time_bonus_scale=80.0,
            centerline_penalty_scale=0.003,
        ),
        StageConfig(
            name="intermediate",
            steps=stage_intermediate_steps,
            track_names=basics_all + intermediate_all,
            randomize_start=True,
            forward_speed_reward_scale=0.0,
            entropy_coef=0.007,
            collision_penalty=800.0,
            lap_time_bonus_scale=60.0,
            centerline_penalty_scale=0.004,
        ),
        StageConfig(
            name="full_grid",
            steps=stage_full_steps,
            track_names=full_grid,
            randomize_start=True,
            forward_speed_reward_scale=0.0,
            entropy_coef=0.004,
            collision_penalty=700.0,
            lap_time_bonus_scale=40.0,
            centerline_penalty_scale=0.002,
        ),
        StageConfig(
            name="race_fast",
            steps=stage_race_fast_steps,
            track_names=full_grid,
            randomize_start=True,
            forward_speed_reward_scale=0.001,
            entropy_coef=0.002,
            collision_penalty=900.0,
            lap_time_bonus_scale=20.0,
            centerline_penalty_scale=0.0,
        ),
    ]


def _latest_json_path(path: str) -> str:
    p = Path(path)
    return str(p.with_name(p.stem + "_latest" + p.suffix))


def peek_checkpoint(path: str) -> Optional[dict]:
    """Return a small summary dict of a .pt checkpoint without constructing a
    trainer. Used by the interactive start menu. Returns None if the file is
    missing or unreadable.
    """
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        state = torch.load(file_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return {
        "total_steps": state.get("total_steps"),
        "updates": state.get("updates"),
        "current_stage_idx": state.get("current_stage_idx"),
        "best_mean_return": state.get("best_mean_return"),
        "obs_dim": state.get("obs_dim"),
        "hidden_size": state.get("hidden_size"),
    }
