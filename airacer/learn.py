"""Simple evolutionary loop for discovering new trainable drivers."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
import os
import pickle
import random
import statistics
import time
from multiprocessing import Pool
from pathlib import Path

from .env import RaceEnv
from .policies import LinearPolicy, MLPPolicy, make_random_policy
from .track import DEFAULT_TRACK_NAME, available_track_names


Policy = LinearPolicy | MLPPolicy


@dataclass
class EpisodeConfig:
    track_name: str
    episode_seed: int
    randomize_start: bool


@dataclass
class CandidateResult:
    policy: Policy
    fitness: float
    forward_progress: float
    collisions: int
    steps: int


@dataclass
class GenerationSummary:
    generation: int
    best_fitness: float
    worst_fitness: float
    mean_fitness: float
    median_fitness: float
    std_fitness: float
    best_forward_progress: float
    mean_forward_progress: float
    best_collisions: int
    best_steps: int
    improvement: float
    seconds: float
    sigma: float


def _run_episode(policy: Policy, max_steps: int, episode: EpisodeConfig):
    env = RaceEnv(
        max_steps=max_steps,
        track_name=episode.track_name,
        fast_observations=True,
    )
    rng = random.Random(episode.episode_seed)
    observation, _ = env.reset(randomize_start=episode.randomize_start, rng=rng)
    total_reward = 0.0
    terminated = False
    truncated = False
    info = {}

    while not terminated and not truncated:
        action = policy.act(observation)
        step_result = env.step(action)
        observation = step_result.observation
        total_reward += step_result.reward
        terminated = step_result.terminated
        truncated = step_result.truncated
        info = step_result.info

    return (
        total_reward,
        info.get("forward_progress", 0.0),
        info.get("collision_count", 0),
        info.get("step_count", 0),
    )


def evaluate_policy(
    policy: Policy,
    max_steps: int,
    episodes: list[EpisodeConfig] | None = None,
):
    if not episodes:
        episodes = [
            EpisodeConfig(
                track_name=DEFAULT_TRACK_NAME,
                episode_seed=0,
                randomize_start=False,
            )
        ]

    fitness_sum = 0.0
    progress_sum = 0.0
    collisions_sum = 0
    steps_sum = 0
    for episode in episodes:
        reward, progress, collisions, steps = _run_episode(policy, max_steps, episode)
        fitness_sum += reward
        progress_sum += progress
        collisions_sum += collisions
        steps_sum += steps

    episodes_count = len(episodes)
    return CandidateResult(
        policy=policy,
        fitness=fitness_sum / episodes_count,
        forward_progress=progress_sum / episodes_count,
        collisions=collisions_sum,
        steps=steps_sum,
    )


def _evaluate_pool(args):
    policy, max_steps, episodes = args
    return evaluate_policy(policy, max_steps, episodes=episodes)


class EvolutionTrainer:
    """Resumable evolutionary trainer for linear and MLP policies."""

    def __init__(
        self,
        population: list[Policy],
        rng: random.Random,
        elite_count: int,
        max_steps: int,
        output_path: str,
        generation: int = 0,
        best_result: CandidateResult | None = None,
        history: list[dict] | None = None,
        training_track_names: list[str] | None = None,
        randomize_start: bool = False,
    ):
        self.population = population
        self.rng = rng
        self.elite_count = elite_count
        self.max_steps = max_steps
        self.output_path = output_path
        self.generation = generation
        self.best_result = best_result
        self.history = history or []
        self.training_track_names = training_track_names or [DEFAULT_TRACK_NAME]
        self.randomize_start = randomize_start

    @classmethod
    def create(
        cls,
        population_size: int = 24,
        elite_count: int = 4,
        max_steps: int = 1800,
        seed: int = 7,
        output_path: str = "artifacts/best_policy.json",
        policy_type: str = "linear",
        hidden_size: int = 16,
        training_track_names: list[str] | None = None,
        randomize_start: bool = False,
    ):
        rng = random.Random(seed)
        observation_size = len(RaceEnv(max_steps=1, fast_observations=True).reset()[0])
        population = [
            make_random_policy(
                policy_type=policy_type,
                observation_size=observation_size,
                rng=rng,
                hidden_size=hidden_size,
            )
            for _ in range(population_size)
        ]
        return cls(
            population=population,
            rng=rng,
            elite_count=elite_count,
            max_steps=max_steps,
            output_path=output_path,
            training_track_names=training_track_names,
            randomize_start=randomize_start,
        )

    def _episode_configs_for_generation(self) -> list[EpisodeConfig]:
        configs: list[EpisodeConfig] = []
        for index, track_name in enumerate(self.training_track_names):
            episode_seed = self.rng.randint(0, 2**31 - 1)
            configs.append(
                EpisodeConfig(
                    track_name=track_name,
                    episode_seed=episode_seed,
                    randomize_start=self.randomize_start,
                )
            )
        return configs

    def run_generation(self, pool: Pool | None = None):
        start = time.perf_counter()
        episodes = self._episode_configs_for_generation()
        args = [(policy, self.max_steps, episodes) for policy in self.population]
        if pool is not None:
            results = pool.map(_evaluate_pool, args)
        else:
            results = [
                evaluate_policy(policy, max_steps=self.max_steps, episodes=episodes)
                for policy in self.population
            ]
        elapsed = time.perf_counter() - start

        results.sort(key=lambda item: item.fitness, reverse=True)
        elites = results[: self.elite_count]

        previous_best = self.best_result.fitness if self.best_result is not None else -math.inf
        improvement = (
            elites[0].fitness - previous_best
            if math.isfinite(previous_best)
            else elites[0].fitness
        )
        if self.best_result is None or elites[0].fitness > self.best_result.fitness:
            self.best_result = elites[0]
            self.best_result.policy.save(self.output_path)
        elites[0].policy.save(_latest_output_path(self.output_path))

        fitness_values = [item.fitness for item in results]
        progress_values = [item.forward_progress for item in results]
        sorted_fitness = sorted(fitness_values)
        median_fitness = sorted_fitness[len(sorted_fitness) // 2]
        std_fitness = statistics.pstdev(fitness_values) if len(fitness_values) > 1 else 0.0
        sigma = max(0.05, 0.22 - self.generation * 0.006)
        summary = GenerationSummary(
            generation=self.generation,
            best_fitness=elites[0].fitness,
            worst_fitness=sorted_fitness[0],
            mean_fitness=sum(fitness_values) / len(fitness_values),
            median_fitness=median_fitness,
            std_fitness=std_fitness,
            best_forward_progress=elites[0].forward_progress,
            mean_forward_progress=sum(progress_values) / len(progress_values),
            best_collisions=elites[0].collisions,
            best_steps=elites[0].steps,
            improvement=improvement,
            seconds=elapsed,
            sigma=sigma,
        )
        self.history.append(summary.__dict__)

        next_population = [elite.policy for elite in elites]
        while len(next_population) < len(self.population):
            parent = self.rng.choice(elites).policy
            next_population.append(parent.mutated(rng=self.rng, sigma=sigma))
        self.population = next_population
        self.generation += 1
        return summary

    def save_state(self, path: str):
        state = {
            "population": self.population,
            "rng_state": self.rng.getstate(),
            "elite_count": self.elite_count,
            "max_steps": self.max_steps,
            "output_path": self.output_path,
            "generation": self.generation,
            "best_result": self.best_result,
            "history": self.history,
            "training_track_names": self.training_track_names,
            "randomize_start": self.randomize_start,
        }
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as handle:
            pickle.dump(state, handle)

    @classmethod
    def load_state(cls, path: str):
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        rng = random.Random()
        rng.setstate(state["rng_state"])
        return cls(
            population=state["population"],
            rng=rng,
            elite_count=state["elite_count"],
            max_steps=state["max_steps"],
            output_path=state["output_path"],
            generation=state["generation"],
            best_result=state["best_result"],
            history=state["history"],
            training_track_names=state.get("training_track_names", [DEFAULT_TRACK_NAME]),
            randomize_start=state.get("randomize_start", False),
        )


def _latest_output_path(output_path: str) -> str:
    path = Path(output_path)
    return str(path.with_name(path.stem + "_latest" + path.suffix))


def _format_summary(summary: GenerationSummary, workers: int, best_ever: float):
    return (
        "gen={gen:>3d} t={secs:5.1f}s w={w} "
        "best={best:8.2f} mean={mean:8.2f} med={median:8.2f} worst={worst:8.2f} std={std:6.2f} "
        "delta={delta:+7.2f} all_time={best_ever:8.2f} "
        "dist={bp:6.1%} pop_dist={mp:6.1%} col={col:>3d} steps={steps:>5d} sigma={sigma:.3f}"
    ).format(
        gen=summary.generation,
        secs=summary.seconds,
        w=workers,
        best=summary.best_fitness,
        mean=summary.mean_fitness,
        median=summary.median_fitness,
        worst=summary.worst_fitness,
        std=summary.std_fitness,
        delta=summary.improvement,
        best_ever=best_ever,
        bp=summary.best_forward_progress,
        mp=summary.mean_forward_progress,
        col=summary.best_collisions,
        steps=summary.best_steps,
        sigma=summary.sigma,
    )


def _format_resume_state(trainer: EvolutionTrainer, state_path: str):
    if not trainer.history:
        return "continuing from %s: no completed generations saved yet" % state_path
    last = trainer.history[-1]
    best_ever = trainer.best_result.fitness if trainer.best_result is not None else last["best_fitness"]
    best_progress = last.get("best_forward_progress", last.get("best_progress", 0.0))
    mean_progress = last.get("mean_forward_progress", last.get("mean_progress", 0.0))
    return (
        "continuing from {path}: last_gen={gen:>3d} "
        "best={best:8.2f} mean={mean:8.2f} dist={bp:6.1%} pop_dist={mp:6.1%} "
        "col={col:>3d} steps={steps:>5d} sigma={sigma:.3f} all_time={all_time:8.2f}"
    ).format(
        path=state_path,
        gen=last["generation"],
        best=last["best_fitness"],
        mean=last["mean_fitness"],
        bp=best_progress,
        mp=mean_progress,
        col=last["best_collisions"],
        steps=last["best_steps"],
        sigma=last["sigma"],
        all_time=best_ever,
    )


def evolve(
    generations: int = 20,
    population_size: int = 24,
    elite_count: int = 4,
    max_steps: int = 1800,
    seed: int = 7,
    output_path: str = "artifacts/best_policy.json",
    workers: int | None = None,
    policy_type: str = "linear",
    hidden_size: int = 16,
    state_path: str | None = None,
    training_track_names: list[str] | None = None,
    randomize_start: bool = False,
):
    if state_path and Path(state_path).exists():
        trainer = EvolutionTrainer.load_state(state_path)
        if training_track_names is not None:
            trainer.training_track_names = training_track_names
        trainer.randomize_start = randomize_start
        print(_format_resume_state(trainer, state_path), flush=True)
    else:
        trainer = EvolutionTrainer.create(
            population_size=population_size,
            elite_count=elite_count,
            max_steps=max_steps,
            seed=seed,
            output_path=output_path,
            policy_type=policy_type,
            hidden_size=hidden_size,
            training_track_names=training_track_names,
            randomize_start=randomize_start,
        )
        print(
            "starting from scratch: policy=%s hidden=%d generations=%d population=%d elites=%d max_steps=%d tracks=%s random_start=%s"
            % (
                policy_type,
                hidden_size,
                generations,
                population_size,
                elite_count,
                max_steps,
                ",".join(trainer.training_track_names),
                randomize_start,
            ),
            flush=True,
        )

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, len(trainer.population))

    print(
        "starting evolution: generations=%d population=%d elites=%d max_steps=%d workers=%d"
        % (generations, len(trainer.population), trainer.elite_count, trainer.max_steps, workers),
        flush=True,
    )

    pool = Pool(workers) if workers > 1 else None
    try:
        total_start = time.perf_counter()
        for _ in range(generations):
            summary = trainer.run_generation(pool=pool)
            best_ever = trainer.best_result.fitness if trainer.best_result else summary.best_fitness
            print(_format_summary(summary, workers, best_ever), flush=True)
            if state_path:
                trainer.save_state(state_path)
        total_elapsed = time.perf_counter() - total_start
        print(
            "done in %.1fs (avg %.2fs/gen)" % (total_elapsed, total_elapsed / max(generations, 1)),
            flush=True,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return trainer.best_result, trainer.history, output_path


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Evolve new aiRacer drivers.")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden-size", type=int, default=16, help="Hidden layer width for MLP policies.")
    parser.add_argument("--output", default="artifacts/best_policy.json")
    parser.add_argument("--state", default=None, help="Path to trainer state pickle. If it exists, training resumes from it.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel worker processes (default: cpu_count - 1)")
    parser.add_argument(
        "--tracks",
        default="all",
        help="Comma-separated track names, or 'all' to train across every registered track.",
    )
    parser.add_argument(
        "--random-start",
        dest="random_start",
        action="store_true",
        help="Randomize episode start position (along centerline with small lateral/heading jitter).",
    )
    parser.add_argument(
        "--no-random-start",
        dest="random_start",
        action="store_false",
        help="Disable randomized start positions.",
    )
    parser.set_defaults(random_start=True)
    args = parser.parse_args(argv)

    if args.tracks.strip().lower() == "all":
        training_track_names = available_track_names()
    else:
        training_track_names = [name.strip() for name in args.tracks.split(",") if name.strip()]

    best, history, output_path = evolve(
        generations=args.generations,
        population_size=args.population,
        elite_count=args.elite,
        max_steps=args.steps,
        seed=args.seed,
        output_path=args.output,
        workers=args.workers,
        policy_type=args.policy_type,
        hidden_size=args.hidden_size,
        state_path=args.state,
        training_track_names=training_track_names,
        randomize_start=args.random_start,
    )

    if best is None:
        raise RuntimeError("No policy was evolved.")

    print(
        "saved=%s fitness=%.2f forward_progress=%.3f collisions=%d"
        % (output_path, best.fitness, best.forward_progress, best.collisions),
        flush=True,
    )
