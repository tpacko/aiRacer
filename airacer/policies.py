"""Policy abstractions and built-in driver implementations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path
from typing import Protocol, Sequence

from . import config
from .neat_policy import load_neat_policy


class DriverPolicy(Protocol):
    """Maps an observation vector to an action tuple."""

    def act(self, observation_vector: Sequence[float]) -> tuple[float, float]:
        ...


class HeuristicPolicy:
    """Simple hand-written baseline using sensors and heading alignment."""

    def act(self, observation_vector: Sequence[float]) -> tuple[float, float]:
        centerline_error = observation_vector[0] * config.SENSOR_RAY_LENGTH
        heading_error = observation_vector[1] * math.pi
        forward_speed = observation_vector[3] * config.CAR_MAX_SPEED
        sensor_count = len(config.SENSOR_RAY_ANGLES)
        sensor_values = observation_vector[4 : 4 + sensor_count]

        left = sensor_values[0] * 0.9 + sensor_values[1] * 0.7 + sensor_values[2] * 0.4
        right = sensor_values[-1] * 0.9 + sensor_values[-2] * 0.7 + sensor_values[-3] * 0.4
        front = sensor_values[len(sensor_values) // 2]
        front_left = sensor_values[2]
        front_right = sensor_values[-3]

        steer_from_heading = heading_error * 1.4
        steer_from_walls = (right - left) * 1.6
        steer_from_front = (front_right - front_left) * 0.8
        steer_from_center = max(-1.0, min(1.0, centerline_error / 120.0)) * 0.15

        steer = steer_from_heading + steer_from_walls + steer_from_front + steer_from_center
        steer = max(-1.0, min(1.0, steer))

        acceleration = 1.0

        if front < 0.22:
            acceleration = -1.0
        elif front < 0.35:
            acceleration = -0.2
        elif abs(steer) > 0.65:
            acceleration = 0.35
        elif abs(steer) > 0.35:
            acceleration = 0.6

        if forward_speed < 15.0 and front < 0.18:
            acceleration = -0.75
            steer = -steer if steer != 0.0 else 0.6

        return steer, acceleration


@dataclass
class LinearPolicy:
    """Small trainable linear policy for evolutionary experiments."""

    steer_weights: list[float]
    acceleration_weights: list[float]
    steer_bias: float = 0.0
    acceleration_bias: float = 0.0

    def act(self, observation_vector: Sequence[float]) -> tuple[float, float]:
        steer = math.tanh(_dot(self.steer_weights, observation_vector) + self.steer_bias)
        acceleration = math.tanh(
            _dot(self.acceleration_weights, observation_vector) + self.acceleration_bias
        )
        return steer, acceleration

    @classmethod
    def random(cls, observation_size: int, rng: random.Random):
        return cls(
            steer_weights=[rng.uniform(-1.0, 1.0) for _ in range(observation_size)],
            acceleration_weights=[rng.uniform(-1.0, 1.0) for _ in range(observation_size)],
            steer_bias=rng.uniform(-0.5, 0.5),
            acceleration_bias=rng.uniform(-0.5, 0.5),
        )

    def mutated(self, rng: random.Random, sigma: float = 0.18):
        return LinearPolicy(
            steer_weights=[w + rng.gauss(0.0, sigma) for w in self.steer_weights],
            acceleration_weights=[w + rng.gauss(0.0, sigma) for w in self.acceleration_weights],
            steer_bias=self.steer_bias + rng.gauss(0.0, sigma),
            acceleration_bias=self.acceleration_bias + rng.gauss(0.0, sigma),
        )

    def to_dict(self):
        return {
            "type": "linear",
            "steer_weights": self.steer_weights,
            "acceleration_weights": self.acceleration_weights,
            "steer_bias": self.steer_bias,
            "acceleration_bias": self.acceleration_bias,
        }

    @classmethod
    def from_dict(cls, data: dict):
        if "acceleration_weights" not in data:
            raise ValueError("Legacy linear policies with separate throttle/brake are no longer supported.")
        return cls(
            steer_weights=list(data["steer_weights"]),
            acceleration_weights=list(data["acceleration_weights"]),
            steer_bias=float(data.get("steer_bias", 0.0)),
            acceleration_bias=float(data.get("acceleration_bias", 0.0)),
        )

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()))


@dataclass
class MLPPolicy:
    """Small multilayer perceptron policy (one hidden layer, tanh).

    ``metadata`` is optional free-form JSON (e.g. produced by the PPO trainer
    to record how many training steps the policy was trained for, which
    curriculum stage it ended in, best mean return, etc.) — it travels with
    the policy file so inference tools can display it without re-deriving.
    """

    w1: list[list[float]]
    b1: list[float]
    w2: list[list[float]]
    b2: list[float]
    metadata: dict | None = None

    def act(self, observation_vector: Sequence[float]) -> tuple[float, float]:
        obs = observation_vector
        hidden = [
            math.tanh(_dot(row, obs) + bias)
            for row, bias in zip(self.w1, self.b1)
        ]
        out = [_dot(row, hidden) + bias for row, bias in zip(self.w2, self.b2)]
        steer = math.tanh(out[0])
        if len(out) >= 3:
            acceleration = _sigmoid(out[1]) - _sigmoid(out[2])
        else:
            acceleration = math.tanh(out[1])
        return steer, acceleration

    @classmethod
    def random(cls, observation_size: int, rng: random.Random, hidden_size: int = 16):
        scale_in = 1.0 / math.sqrt(observation_size)
        scale_hidden = 1.0 / math.sqrt(hidden_size)
        w1 = [
            [rng.gauss(0.0, scale_in) for _ in range(observation_size)]
            for _ in range(hidden_size)
        ]
        b1 = [rng.gauss(0.0, 0.1) for _ in range(hidden_size)]
        w2 = [
            [rng.gauss(0.0, scale_hidden) for _ in range(hidden_size)]
            for _ in range(2)
        ]
        b2 = [rng.gauss(0.0, 0.1) for _ in range(2)]
        return cls(w1=w1, b1=b1, w2=w2, b2=b2)

    def mutated(self, rng: random.Random, sigma: float = 0.1):
        return MLPPolicy(
            w1=[[w + rng.gauss(0.0, sigma) for w in row] for row in self.w1],
            b1=[b + rng.gauss(0.0, sigma) for b in self.b1],
            w2=[[w + rng.gauss(0.0, sigma) for w in row] for row in self.w2],
            b2=[b + rng.gauss(0.0, sigma) for b in self.b2],
        )

    def to_dict(self):
        data: dict = {
            "type": "mlp",
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            w1=[list(row) for row in data["w1"]],
            b1=list(data["b1"]),
            w2=[list(row) for row in data["w2"]],
            b2=list(data["b2"]),
            metadata=data.get("metadata"),
        )

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()))


def load_policy_from_file(path: str | Path):
    if Path(path).suffix.lower() in (".pkl", ".pickle"):
        return load_neat_policy(path)
    data = json.loads(Path(path).read_text())
    policy_type = data.get("type", "linear")
    if policy_type == "linear":
        return LinearPolicy.from_dict(data)
    if policy_type == "mlp":
        return MLPPolicy.from_dict(data)
    if policy_type == "neat":
        return load_neat_policy(path)
    raise ValueError("Unknown policy type in %s: %s" % (path, policy_type))


def make_random_policy(policy_type: str, observation_size: int, rng: random.Random, hidden_size: int = 16):
    policy_type = policy_type.lower()
    if policy_type == "linear":
        return LinearPolicy.random(observation_size=observation_size, rng=rng)
    if policy_type == "mlp":
        return MLPPolicy.random(observation_size=observation_size, rng=rng, hidden_size=hidden_size)
    raise ValueError("Unknown policy type: %s" % policy_type)


def create_policy(name: str, policy_path: str | None = None):
    normalized = name.lower()
    if normalized == "scripted":
        return HeuristicPolicy()
    if normalized in ("linear", "mlp", "neat", "policy"):
        if not policy_path:
            raise ValueError("The %s driver requires a policy file path." % normalized)
        return load_policy_from_file(policy_path)
    raise ValueError("Unknown driver policy: %s" % name)


def _dot(weights: Sequence[float], values: Sequence[float]):
    return sum(weight * value for weight, value in zip(weights, values))


def _sigmoid(value: float):
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
