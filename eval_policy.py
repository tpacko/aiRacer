"""Evaluate a saved policy on held-out (track, random-start) conditions."""

from __future__ import annotations

import argparse
import random

from airacer.env import RaceEnv
from airacer.policies import load_policy_from_file
from airacer.track import available_track_names


def _run_single(policy, max_steps, track_name, seed):
    env = RaceEnv(max_steps=max_steps, track_name=track_name)
    rng = random.Random(seed)
    observation, _ = env.reset(randomize_start=True, rng=rng)
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
    return {
        "reward": total_reward,
        "forward_progress": info.get("forward_progress", 0.0),
        "collision_count": info.get("collision_count", 0),
        "step_count": info.get("step_count", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved policy across tracks / random starts.")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--tracks", default="all")
    parser.add_argument("--episodes-per-track", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=9000)
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()

    if args.tracks.strip().lower() == "all":
        track_names = available_track_names()
    else:
        track_names = [n.strip() for n in args.tracks.split(",") if n.strip()]

    policy = load_policy_from_file(args.policy)

    overall_progress = []
    overall_reward = []
    per_track_summary = []
    for track_name in track_names:
        per_track_progress = []
        per_track_reward = []
        for i in range(args.episodes_per_track):
            seed = args.base_seed + i + 997 * hash(track_name) % 999983
            result = _run_single(policy, args.steps, track_name, seed)
            per_track_progress.append(result["forward_progress"])
            per_track_reward.append(result["reward"])
        mean_progress = sum(per_track_progress) / len(per_track_progress)
        mean_reward = sum(per_track_reward) / len(per_track_reward)
        min_progress = min(per_track_progress)
        max_progress = max(per_track_progress)
        per_track_summary.append(
            (track_name, mean_progress, min_progress, max_progress, mean_reward)
        )
        overall_progress.extend(per_track_progress)
        overall_reward.extend(per_track_reward)
        print(
            "track=%-16s mean_dist=%5.1f%% min=%5.1f%% max=%5.1f%% mean_reward=%8.2f"
            % (
                track_name,
                mean_progress * 100.0,
                min_progress * 100.0,
                max_progress * 100.0,
                mean_reward,
            )
        )
    overall_mean = sum(overall_progress) / len(overall_progress)
    overall_min = min(overall_progress)
    print(
        "OVERALL          mean_dist=%5.1f%% worst=%5.1f%% mean_reward=%8.2f n=%d"
        % (
            overall_mean * 100.0,
            overall_min * 100.0,
            sum(overall_reward) / len(overall_reward),
            len(overall_progress),
        )
    )


if __name__ == "__main__":
    main()
