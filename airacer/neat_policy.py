"""NEAT policy loading, execution, and training helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import functools
from multiprocessing import Pool
import pickle
from pathlib import Path
import re
from textwrap import dedent
from typing import Any, Sequence

from .env import RaceEnv


def _load_neat():
    try:
        import neat
    except ImportError as exc:
        raise RuntimeError(
            "neat-python is required for NEAT policies. Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return neat


@dataclass
class NeatPolicy:
    """Executable NEAT feedforward network policy."""

    network: Any

    def act(self, observation_vector: Sequence[float]) -> tuple[float, float]:
        output = list(self.network.activate(list(observation_vector)))
        if len(output) < 2:
            raise ValueError("NEAT policy must produce at least 2 outputs: steer and acceleration.")
        steer = max(-1.0, min(1.0, float(output[0])))
        acceleration = max(-1.0, min(1.0, float(output[1])))
        return steer, acceleration


@dataclass
class EvalResult:
    fitness: float
    total_reward: float
    forward_progress: float
    monotonic_progress: float
    checkpoints_hit: int
    laps_completed: int
    collisions: int
    steps: int
    avg_speed: float
    max_speed: float
    avg_forward_speed: float
    steer_fluency: float
    lap_time_avg: float
    lap_time_best: float
    stuck_steps: int
    truncated: bool


def default_neat_config_text(input_size: int, output_size: int = 2):
    return dedent(
        f"""
        [NEAT]
        fitness_criterion     = max
        fitness_threshold     = 100000.0
        pop_size              = 150
        reset_on_extinction   = True

        [DefaultGenome]
        activation_default      = tanh
        activation_mutate_rate  = 0.0
        activation_options      = tanh

        aggregation_default     = sum
        aggregation_mutate_rate = 0.0
        aggregation_options     = sum

        bias_init_mean          = 0.0
        bias_init_stdev         = 1.0
        bias_max_value          = 30.0
        bias_min_value          = -30.0
        bias_mutate_power       = 0.5
        bias_mutate_rate        = 0.7
        bias_replace_rate       = 0.1

        compatibility_disjoint_coefficient = 1.0
        compatibility_weight_coefficient   = 0.6

        conn_add_prob           = 0.5
        conn_delete_prob        = 0.3

        enabled_default         = True
        enabled_mutate_rate     = 0.02

        feed_forward            = True
        initial_connection      = full_direct

        node_add_prob           = 0.15
        node_delete_prob        = 0.12

        num_hidden              = 0
        num_inputs              = {input_size}
        num_outputs             = {output_size}

        response_init_mean      = 1.0
        response_init_stdev     = 0.0
        response_max_value      = 30.0
        response_min_value      = -30.0
        response_mutate_power   = 0.0
        response_mutate_rate    = 0.0
        response_replace_rate   = 0.0

        weight_init_mean        = 0.0
        weight_init_stdev       = 1.0
        weight_max_value        = 30
        weight_min_value        = -30
        weight_mutate_power     = 0.5
        weight_mutate_rate      = 0.8
        weight_replace_rate     = 0.1

        [DefaultSpeciesSet]
        compatibility_threshold = 2.2

        [DefaultStagnation]
        species_fitness_func = max
        max_stagnation       = 20
        species_elitism      = 1

        [DefaultReproduction]
        elitism            = 2
        survival_threshold = 0.25
        """
    ).strip() + "\n"


def write_default_neat_config(path: str | Path):
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    input_size = len(RaceEnv(max_steps=1, fast_observations=True).reset()[0])
    config_path.write_text(default_neat_config_text(input_size=input_size), encoding="utf-8")
    return config_path


def _config_num_inputs(path: str | Path) -> int | None:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^\s*num_inputs\s*=\s*(\d+)", text, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def save_neat_policy(path: str | Path, genome: Any, config_text: str):
    payload = {
        "type": "neat",
        "config_text": config_text,
        "genome": genome,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(payload, handle)


def _format_lap(seconds: float) -> str:
    if seconds <= 0.0:
        return "  --:--"
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return "%d:%05.2f" % (minutes, remainder)


def _read_saved_genome_fitness(path: str | Path) -> float | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    genome = payload.get("genome") if isinstance(payload, dict) else None
    fitness = getattr(genome, "fitness", None)
    if fitness is None:
        return None
    try:
        return float(fitness)
    except (TypeError, ValueError):
        return None


class BestGenomeSaver:
    """Reporter that persists the best-so-far genome each generation."""

    def __init__(self, output_path: str | Path, config_text: str):
        self.output_path = output_path
        self.config_text = config_text
        self.best_fitness = _read_saved_genome_fitness(output_path)
        self.last_generation = None
        if self.best_fitness is not None:
            print("best-genome-saver: existing best fitness=%.4f at %s" % (self.best_fitness, output_path), flush=True)

    def start_generation(self, generation):
        self.last_generation = generation

    def end_generation(self, config, population, species_set):
        return None

    def post_evaluate(self, config, population, species, best_genome):
        metrics = getattr(best_genome, "eval_metrics", None)
        evaluated = [genome.eval_metrics for genome in population.values() if hasattr(genome, "eval_metrics")]
        if metrics is not None and evaluated:
            def _mean(attr):
                return sum(getattr(m, attr) for m in evaluated) / len(evaluated)
            block = (
                "\n"
                "==== gen {gen} =================================================================\n"
                "  BEST  fit={b_fit:>9.2f}  dist={b_dist:>6.1f}%  laps={b_laps}  cps={b_cps:>3d}  col={b_col:>3d}  steps={b_steps:>5d}\n"
                "        avg_spd={b_avg:>5.0f}  max_spd={b_max:>5.0f}  fwd_spd={b_fwd:>5.0f}  steer_fluency={b_sf:.3f}  stuck={b_stuck:>4d}\n"
                "        lap_avg={b_la}  lap_best={b_lb}  truncated={b_tr}\n"
                "  MEAN  fit={m_fit:>9.2f}  dist={m_dist:>6.1f}%  laps={m_laps:>4.1f}  cps={m_cps:>5.1f}  col={m_col:>5.1f}  steps={m_steps:>7.1f}\n"
                "        avg_spd={m_avg:>5.0f}  max_spd={m_max:>5.0f}  fwd_spd={m_fwd:>5.0f}  steer_fluency={m_sf:.3f}  stuck={m_stuck:>5.1f}\n"
            ).format(
                gen=self.last_generation,
                b_fit=metrics.fitness,
                b_dist=metrics.forward_progress * 100.0,
                b_laps=metrics.laps_completed,
                b_cps=metrics.checkpoints_hit,
                b_col=metrics.collisions,
                b_steps=metrics.steps,
                b_avg=metrics.avg_speed,
                b_max=metrics.max_speed,
                b_fwd=metrics.avg_forward_speed,
                b_sf=metrics.steer_fluency,
                b_stuck=metrics.stuck_steps,
                b_la=_format_lap(metrics.lap_time_avg),
                b_lb=_format_lap(metrics.lap_time_best),
                b_tr="yes" if metrics.truncated else "no",
                m_fit=_mean("fitness"),
                m_dist=_mean("forward_progress") * 100.0,
                m_laps=_mean("laps_completed"),
                m_cps=_mean("checkpoints_hit"),
                m_col=_mean("collisions"),
                m_steps=_mean("steps"),
                m_avg=_mean("avg_speed"),
                m_max=_mean("max_speed"),
                m_fwd=_mean("avg_forward_speed"),
                m_sf=_mean("steer_fluency"),
                m_stuck=_mean("stuck_steps"),
            )
            print(block, flush=True)
        if self.best_fitness is None or best_genome.fitness > self.best_fitness:
            prior = self.best_fitness
            self.best_fitness = best_genome.fitness
            save_neat_policy(self.output_path, genome=best_genome, config_text=self.config_text)
            print(
                "saved new best: gen=%s fitness=%.4f (prior=%s) dist=%.1f%% col=%s path=%s"
                % (
                    self.last_generation,
                    best_genome.fitness,
                    "%.4f" % prior if prior is not None else "none",
                    (metrics.forward_progress * 100.0) if metrics is not None else float("nan"),
                    metrics.collisions if metrics is not None else "?",
                    self.output_path,
                ),
                flush=True,
            )

    def post_reproduction(self, config, population, species):
        return None

    def complete_extinction(self):
        return None

    def found_solution(self, config, generation, best):
        if self.best_fitness is None or best.fitness >= self.best_fitness:
            self.best_fitness = best.fitness
            save_neat_policy(self.output_path, genome=best, config_text=self.config_text)

    def species_stagnant(self, sid, species):
        return None

    def info(self, msg):
        return None


def load_neat_policy(path: str | Path):
    neat = _load_neat()
    source = Path(path)
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("type") != "neat":
        raise ValueError("Unknown NEAT policy bundle format in %s" % source)
    config_text = payload["config_text"]
    config_path = source.with_suffix(".config.tmp")
    config_path.write_text(config_text, encoding="utf-8")
    try:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )
    finally:
        config_path.unlink(missing_ok=True)
    network = neat.nn.FeedForwardNetwork.create(payload["genome"], config)
    return NeatPolicy(network=network)


def _evaluate_genomes(genomes, config, max_steps: int):
    for _, genome in genomes:
        result = _evaluate_genome(genome, config, max_steps)
        genome.fitness = result.fitness
        genome.eval_metrics = result


def _evaluate_genome(genome, config, max_steps: int):
    neat = _load_neat()
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    policy = NeatPolicy(network=network)
    env = RaceEnv(max_steps=max_steps, fast_observations=True)
    observation, _ = env.reset()
    total_reward = 0.0
    terminated = False
    truncated = False
    info = {}
    speed_sum = 0.0
    fwd_speed_sum = 0.0
    max_speed = 0.0
    steer_delta_sum = 0.0
    prev_steer = 0.0
    lap_times = []
    last_lap_step = 0
    low_speed_steps = 0
    while not terminated and not truncated:
        action = policy.act(observation)
        step_result = env.step(action)
        observation = step_result.observation
        total_reward += step_result.reward
        terminated = step_result.terminated
        truncated = step_result.truncated
        info = step_result.info
        speed = info["speed"]
        fwd = info["forward_speed"]
        speed_sum += speed
        fwd_speed_sum += fwd
        if speed > max_speed:
            max_speed = speed
        steer_delta_sum += abs(action[0] - prev_steer)
        prev_steer = action[0]
        if info["forward_speed"] < 18.0:
            low_speed_steps += 1
        if info.get("lap_completed"):
            lap_times.append(info["step_count"] - last_lap_step)
            last_lap_step = info["step_count"]
    steps = info.get("step_count", 0) or 1
    dt = 1.0 / max(1, env.simulation_hz)
    lap_time_avg = (sum(lap_times) * dt) / len(lap_times) if lap_times else 0.0
    lap_time_best = (min(lap_times) * dt) if lap_times else 0.0
    return EvalResult(
        fitness=total_reward,
        total_reward=total_reward,
        forward_progress=info["forward_progress"],
        monotonic_progress=info.get("monotonic_progress", 0.0),
        checkpoints_hit=info.get("checkpoints_hit", 0),
        laps_completed=info.get("completed_laps", 0),
        collisions=info["collision_count"],
        steps=steps,
        avg_speed=speed_sum / steps,
        max_speed=max_speed,
        avg_forward_speed=fwd_speed_sum / steps,
        steer_fluency=steer_delta_sum / steps,
        lap_time_avg=lap_time_avg,
        lap_time_best=lap_time_best,
        stuck_steps=low_speed_steps,
        truncated=truncated,
    )


def _evaluate_genome_worker(args):
    genome, config, max_steps = args
    return _evaluate_genome(genome, config, max_steps)


def _evaluate_population(genomes, config, max_steps: int, workers: int, pool: Pool | None = None):
    if pool is not None:
        args = [(genome, config, max_steps) for _, genome in genomes]
        results = pool.map(_evaluate_genome_worker, args)
        for (_, genome), result in zip(genomes, results):
            genome.fitness = result.fitness
            genome.eval_metrics = result
        return
    if workers > 1:
        args = [(genome, config, max_steps) for _, genome in genomes]
        with Pool(workers) as local_pool:
            results = local_pool.map(_evaluate_genome_worker, args)
        for (_, genome), result in zip(genomes, results):
            genome.fitness = result.fitness
            genome.eval_metrics = result
        return
    _evaluate_genomes(genomes, config, max_steps)


def _find_latest_checkpoint(checkpoint_prefix: str):
    prefix_path = Path(checkpoint_prefix)
    parent = prefix_path.parent if prefix_path.parent != Path("") else Path(".")
    prefix_name = prefix_path.name
    if not parent.exists():
        return None

    latest_path = None
    latest_generation = -1
    pattern = re.compile(r"^%s(\d+)$" % re.escape(prefix_name))
    for candidate in parent.iterdir():
        match = pattern.match(candidate.name)
        if not match:
            continue
        generation = int(match.group(1))
        if generation > latest_generation:
            latest_generation = generation
            latest_path = candidate
    return latest_path


def train_neat(
    generations: int,
    config_path: str | Path,
    output_path: str | Path,
    max_steps: int,
    workers: int = 1,
    checkpoint_prefix: str = "artifacts/neat-checkpoint-",
    restore_checkpoint: str | None = None,
):
    neat = _load_neat()
    config_text = Path(config_path).read_text(encoding="utf-8")
    if restore_checkpoint is None:
        latest_checkpoint = _find_latest_checkpoint(checkpoint_prefix)
        if latest_checkpoint is not None:
            restore_checkpoint = str(latest_checkpoint)
    if restore_checkpoint:
        population = neat.Checkpointer.restore_checkpoint(restore_checkpoint)
        config = population.config
        print("continuing from checkpoint=%s" % restore_checkpoint, flush=True)
    else:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )
        population = neat.Population(config)
        print("starting from scratch", flush=True)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(BestGenomeSaver(output_path=output_path, config_text=config_text))
    population.add_reporter(
        neat.Checkpointer(generation_interval=1, filename_prefix=checkpoint_prefix)
    )
    pool = Pool(workers) if workers > 1 else None
    try:
        winner = population.run(
            functools.partial(_evaluate_population, max_steps=max_steps, workers=workers, pool=pool),
            generations,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    prior_best = _read_saved_genome_fitness(output_path)
    winner_fitness = getattr(winner, "fitness", None)
    if prior_best is None or (winner_fitness is not None and winner_fitness > prior_best):
        save_neat_policy(output_path, genome=winner, config_text=config_text)
        print("saved winner fitness=%.4f (prior best=%s)" % (winner_fitness, prior_best), flush=True)
    else:
        print("kept prior best fitness=%.4f (winner=%.4f)" % (prior_best, winner_fitness or float("-inf")), flush=True)
    return winner, output_path


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train a NEAT policy for aiRacer.")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--steps", type=int, default=1900)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--config", default="artifacts/neat_config.ini")
    parser.add_argument("--output", default="artifacts/best_neat_policy.pkl")
    parser.add_argument("--checkpoint-prefix", default="artifacts/neat-checkpoint-")
    parser.add_argument("--resume", default=None, help="Checkpoint file to resume from.")
    parser.add_argument("--write-config", action="store_true", help="Write a default NEAT config file and exit.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    expected_inputs = len(RaceEnv(max_steps=1, fast_observations=True).reset()[0])
    existing_inputs = _config_num_inputs(config_path) if config_path.exists() else None
    inputs_changed = existing_inputs is not None and existing_inputs != expected_inputs
    if args.write_config or not config_path.exists() or inputs_changed:
        write_default_neat_config(config_path)
        if inputs_changed:
            print(
                "observation size changed (%d -> %d); regenerated %s and discarding stale checkpoints"
                % (existing_inputs, expected_inputs, config_path),
                flush=True,
            )
        if args.write_config:
            print("wrote default config to %s" % config_path, flush=True)
            return

    restore_checkpoint = args.resume
    if inputs_changed:
        prefix_path = Path(args.checkpoint_prefix)
        parent = prefix_path.parent if str(prefix_path.parent) != "" else Path(".")
        if parent.exists():
            pattern = re.compile(r"^%s\d+$" % re.escape(prefix_path.name))
            for candidate in parent.iterdir():
                if pattern.match(candidate.name):
                    candidate.unlink()
        restore_checkpoint = None

    _, output_path = train_neat(
        generations=args.generations,
        config_path=config_path,
        output_path=args.output,
        max_steps=args.steps,
        workers=args.workers,
        checkpoint_prefix=args.checkpoint_prefix,
        restore_checkpoint=restore_checkpoint,
    )
    print("saved=%s" % output_path, flush=True)
