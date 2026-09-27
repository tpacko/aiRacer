"""Entrypoint for PPO reinforcement-learning training.

On launch, a small interactive menu asks where to start from. Pick a number:

  0 = resume the existing curriculum checkpoint (continues where it left off)
  1..N = warm-start from the baseline network and begin curriculum stage N
         (fresh optimizer, step counter advanced to that stage's start — use
         this to re-train a specific stage "from scratch" using the
         prelearned actor as initialization; you will be asked whether to
         transfer the critic too)
  C = warm-start from the current curriculum checkpoint and jump to a stage
  R = train from random-initialized weights, stage 1

All hyperparameters are constants in this file; edit and re-run.

Inspect the trained policy with the existing runners once training saves JSON:

    python scripted_main.py --driver policy --policy artifacts/ppo_curriculum_policy.json
or
    python scripted_main.py --driver policy --policy artifacts/ppo_curriculum_policy_latest.json
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from airacer.rl import (
    PPOTrainer,
    RLConfig,
    make_progressive_stages,
    peek_checkpoint,
)


# -----------------------------------------------------------------------------
# Progressive curriculum (policy-steps per stage; a "policy step" = frame_skip
# sim steps). See airacer.rl.make_progressive_stages for detail.
# -----------------------------------------------------------------------------
STAGE_LENGTH_MULTIPLIER = 5
STAGE_BASICS_STEPS = 300_000 * STAGE_LENGTH_MULTIPLIER          # 2 basic short tracks, fwd only, fixed start
STAGE_BASICS_REVERSE_STEPS = 400_000 * STAGE_LENGTH_MULTIPLIER  # basics + reversed variants (both turn dirs)
STAGE_INTERMEDIATE_STEPS = 600_000 * STAGE_LENGTH_MULTIPLIER    # + 2 intermediate tracks, both directions
STAGE_FULL_GRID_STEPS = 800_000 * STAGE_LENGTH_MULTIPLIER       # + 3 advanced tracks, both directions
STAGE_RACE_FAST_STEPS = 1_400_000 * STAGE_LENGTH_MULTIPLIER       # same full grid, lower entropy
TOTAL_STEPS = (
    STAGE_BASICS_STEPS
    + STAGE_BASICS_REVERSE_STEPS
    + STAGE_INTERMEDIATE_STEPS
    + STAGE_FULL_GRID_STEPS
    + STAGE_RACE_FAST_STEPS
)

# -----------------------------------------------------------------------------
# Rollout / PPO hyperparameters
# -----------------------------------------------------------------------------
ROLLOUT_STEPS = 2048
ROLLOUT_WORKERS = 4
FRAME_SKIP = 4
N_EPOCHS = 10
MINIBATCH_SIZE = 256
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
VALUE_CLIP_RANGE = 100.0
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
LEARNING_RATE = 3.0e-4
WARM_START_LEARNING_RATE = 5.0e-4
INITIAL_LOG_STD = -0.5

# When warm-starting from an already-trained actor, keep the learned mean policy
# but reset PPO exploration noise. The old checkpoint may have large log_std
# values that are useful for broad exploration but too crash-prone for careful
# relearning. Values are log(sigma): -1.1 ~= 0.33 steer, -0.7 ~= 0.50 accel.
WARM_START_RESET_LOG_STD = (-1.1, -0.7)

# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------
HIDDEN_SIZE = 128

# -----------------------------------------------------------------------------
# Env
# -----------------------------------------------------------------------------
MAX_STEPS_PER_EPISODE = 2400

# -----------------------------------------------------------------------------
# Saving & logging
# -----------------------------------------------------------------------------
# New curriculum writes to its own paths so the original baseline at
# ``artifacts/ppo_state.pt`` / ``artifacts/ppo_policy.json`` stays intact and
# can keep serving as a warm-start source.
CHECKPOINT_PATH = "artifacts/ppo_curriculum_state.pt"
POLICY_JSON_PATH = "artifacts/ppo_curriculum_policy.json"
PPO_LOG_PATH = "artifacts/ppo_curriculum_metrics.jsonl"
PRINT_EVERY_UPDATES = 1
RECENT_EPISODE_WINDOW = 20

# When warm-starting actor-only, run a critic-only prelearning phase before PPO.
# The actor generates rollouts; only the critic is updated so the first PPO
# updates start with a useful value baseline instead of a random one.
CRITIC_PRETRAIN_UPDATES = 30
CRITIC_PRETRAIN_EPOCHS = 8

# Warm-start relearning profile: keep the learned actor's useful line-following
# behavior, but make wall contact expensive and exploration small until clean
# driving is stable. Random-init runs still use the broader exploration profile
# from make_progressive_stages.
USE_WARM_START_RELEARN_PROFILE = True
WARM_START_STAGE_OVERRIDES = {
    "basics_forward": {
        "entropy_coef": 0.001,
        "collision_penalty": 1200.0,
        "lap_time_bonus_scale": 20.0,
        "centerline_penalty_scale": 0.006,
        "forward_speed_reward_scale": 0.0,
    },
    "basics_both_directions": {
        "entropy_coef": 0.001,
        "collision_penalty": 1200.0,
        "lap_time_bonus_scale": 20.0,
        "centerline_penalty_scale": 0.006,
        "forward_speed_reward_scale": 0.0,
    },
    "intermediate": {
        "entropy_coef": 0.0015,
        "collision_penalty": 1000.0,
        "lap_time_bonus_scale": 20.0,
        "centerline_penalty_scale": 0.005,
        "forward_speed_reward_scale": 0.0,
    },
    "full_grid": {
        "entropy_coef": 0.0015,
        "collision_penalty": 900.0,
        "lap_time_bonus_scale": 20.0,
        "centerline_penalty_scale": 0.003,
        "forward_speed_reward_scale": 0.0,
    },
    "race_fast": {
        "entropy_coef": 0.001,
        "collision_penalty": 700.0,
        "lap_time_bonus_scale": 20.0,
        "centerline_penalty_scale": 0.0,
        "forward_speed_reward_scale": 0.0,
    },
}

# -----------------------------------------------------------------------------
# Baseline weights used for warm-starts (the pre-curriculum PPO policy)
# -----------------------------------------------------------------------------
BASELINE_CHECKPOINT = "artifacts/ppo_state.pt"

SEED = 42


# -----------------------------------------------------------------------------
# Interactive start menu
# -----------------------------------------------------------------------------

def _fmt_step(n):
    return f"{int(n):,}" if isinstance(n, (int, float)) else str(n)


def _format_stage_label(stage_index: int, stages):
    stage = stages[stage_index]
    return f"stage {stage_index + 1}/{len(stages)} [{stage.name}] — {len(stage.track_names)} tracks"


def _make_warm_start_relearn_stages(stages):
    if not USE_WARM_START_RELEARN_PROFILE:
        return stages
    tuned = []
    for stage in stages:
        overrides = WARM_START_STAGE_OVERRIDES.get(stage.name)
        if overrides is None:
            tuned.append(stage)
            continue
        tuned.append(replace(stage, name="careful_" + stage.name, **overrides))
    return tuned


def _prompt_choice(stages) -> dict:
    """Build and render the start menu, read the user's choice, and return a
    small instruction dict understood by ``main()``.

    Returned dict keys:
      - "mode": "resume" | "warm_start" | "random"
      - "source": path to checkpoint (for warm_start / resume)
      - "stage_index": target stage (0-based) for warm_start / random
    """
    baseline_info = peek_checkpoint(BASELINE_CHECKPOINT)
    curriculum_info = peek_checkpoint(CHECKPOINT_PATH)

    bar = "=" * 72
    print("")
    print(bar)
    print(" PPO training — select starting point")
    print(bar)

    options: list[tuple[str, str, dict | None]] = []

    # 0 — resume existing curriculum checkpoint
    if curriculum_info is not None:
        stage_idx = curriculum_info.get("current_stage_idx") or 0
        stage_label = "?"
        if 0 <= stage_idx < len(stages):
            stage_label = _format_stage_label(stage_idx, stages)
        desc = (
            f"Resume curriculum checkpoint  "
            f"(steps={_fmt_step(curriculum_info.get('total_steps', 0))}, "
            f"updates={_fmt_step(curriculum_info.get('updates', 0))}, {stage_label})"
        )
        options.append(("0", desc, {"mode": "resume", "source": CHECKPOINT_PATH}))
    else:
        options.append((
            "0",
            f"Resume curriculum checkpoint  (unavailable — {CHECKPOINT_PATH} not found)",
            None,
        ))

    # 1..N — warm-start from BASELINE at stage N
    baseline_ok = baseline_info is not None
    for idx, stage in enumerate(stages):
        key = str(idx + 1)
        if baseline_ok:
            desc = (
                f"Warm-start from baseline, begin at {_format_stage_label(idx, stages)}"
            )
            payload = {
                "mode": "warm_start",
                "source": BASELINE_CHECKPOINT,
                "stage_index": idx,
            }
        else:
            desc = f"Warm-start from baseline  (unavailable — {BASELINE_CHECKPOINT} not found)"
            payload = None
        options.append((key, desc, payload))

    # C — warm-start from CURRICULUM checkpoint at a chosen stage
    if curriculum_info is not None:
        options.append((
            "C",
            "Warm-start from current curriculum checkpoint (you'll then pick a stage)",
            {"mode": "warm_start_prompt_stage", "source": CHECKPOINT_PATH},
        ))
    else:
        options.append((
            "C",
            f"Warm-start from curriculum checkpoint  (unavailable — {CHECKPOINT_PATH} not found)",
            None,
        ))

    # R — random init, stage 1
    options.append((
        "R",
        "Random init, begin at stage 1",
        {"mode": "random", "stage_index": 0},
    ))

    options.append(("Q", "Quit without training", {"mode": "quit"}))

    # Render
    for key, desc, payload in options:
        marker = " " if payload is None else "*"
        print(f"  [{key}] {marker} {desc}")
    print(bar)
    print(" (letter shortcuts are case-insensitive)")

    while True:
        raw = input(" choice: ").strip()
        if not raw:
            continue
        choice_key = raw.upper()
        match = next(
            (opt for opt in options if opt[0].upper() == choice_key and opt[2] is not None),
            None,
        )
        if match is None:
            print("  ... not a valid choice, try again.")
            continue
        payload = match[2]
        if payload.get("mode") == "warm_start_prompt_stage":
            stage_idx = _prompt_stage_index(stages)
            return {
                "mode": "warm_start",
                "source": payload["source"],
                "stage_index": stage_idx,
            }
        return payload


def _prompt_stage_index(stages) -> int:
    bar = "-" * 72
    print("")
    print(bar)
    print(" pick target stage:")
    for idx, stage in enumerate(stages):
        print(f"  [{idx + 1}] {_format_stage_label(idx, stages)}")
    print(bar)
    while True:
        raw = input(" stage: ").strip()
        if not raw.isdigit():
            print("  ... enter a stage number.")
            continue
        idx = int(raw) - 1
        if not (0 <= idx < len(stages)):
            print(f"  ... out of range (1..{len(stages)}).")
            continue
        return idx


def _prompt_transfer_critic() -> bool:
    bar = "-" * 72
    print("")
    print(bar)
    print(" critic transfer")
    print(" Warm-start always transfers the actor NN.")
    print(" Transfer the critic NN as well?")
    print("   [Y] yes - reuse actor and critic from checkpoint")
    print("   [N] no  - reuse actor only, then prelearn critic from actor rollouts")
    print(" Default is N because the careful relearning reward changes critic targets.")
    print(bar)
    while True:
        raw = input(" transfer critic? [y/N]: ").strip().lower()
        if raw in ("", "n", "no"):
            return False
        if raw in ("y", "yes"):
            return True
        print("  ... enter Y or N.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    base_stages = make_progressive_stages(
        stage_basics_steps=STAGE_BASICS_STEPS,
        stage_basics_reverse_steps=STAGE_BASICS_REVERSE_STEPS,
        stage_intermediate_steps=STAGE_INTERMEDIATE_STEPS,
        stage_full_steps=STAGE_FULL_GRID_STEPS,
        stage_race_fast_steps=STAGE_RACE_FAST_STEPS,
    )

    choice = _prompt_choice(base_stages)
    if choice.get("mode") == "quit":
        print("aborted.")
        return

    mode = choice["mode"]
    stages = (
        _make_warm_start_relearn_stages(base_stages)
        if mode == "warm_start"
        else base_stages
    )
    learning_rate = WARM_START_LEARNING_RATE if mode == "warm_start" else LEARNING_RATE

    cfg = RLConfig(
        hidden_size=HIDDEN_SIZE,
        initial_log_std=INITIAL_LOG_STD,
        total_steps=TOTAL_STEPS,
        rollout_steps=ROLLOUT_STEPS,
        rollout_workers=ROLLOUT_WORKERS,
        frame_skip=FRAME_SKIP,
        n_epochs=N_EPOCHS,
        minibatch_size=MINIBATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        value_clip_range=VALUE_CLIP_RANGE,
        value_coef=VALUE_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        learning_rate=learning_rate,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        stages=stages,
        checkpoint_path=CHECKPOINT_PATH,
        policy_json_path=POLICY_JSON_PATH,
        ppo_log_path=PPO_LOG_PATH,
        print_every_updates=PRINT_EVERY_UPDATES,
        recent_episode_window=RECENT_EPISODE_WINDOW,
        critic_pretrain_updates=CRITIC_PRETRAIN_UPDATES,
        critic_pretrain_epochs=CRITIC_PRETRAIN_EPOCHS,
        seed=SEED,
    )
    trainer = PPOTrainer(cfg)

    if mode == "resume":
        print(f"Resuming from {choice['source']}", flush=True)
        trainer.load_checkpoint(choice["source"])
    elif mode == "warm_start":
        transfer_critic = _prompt_transfer_critic()
        print(
            f"Warm-starting from {choice['source']}, jumping to stage "
            f"{choice['stage_index'] + 1}, critic_transfer={transfer_critic}, "
            f"careful_relearn={USE_WARM_START_RELEARN_PROFILE}",
            flush=True,
        )
        trainer.warm_start_from_checkpoint(
            choice["source"],
            transfer_critic=transfer_critic,
            reset_log_std=WARM_START_RESET_LOG_STD,
        )
        if choice["stage_index"] != 0:
            trainer.jump_to_stage(choice["stage_index"])
        if not transfer_critic:
            trainer.pretrain_critic()
    elif mode == "random":
        if choice.get("stage_index", 0) != 0:
            trainer.jump_to_stage(choice["stage_index"])
        print("Training from random-initialized weights.", flush=True)
    else:
        raise RuntimeError(f"unknown mode: {mode}")

    trainer.train()


if __name__ == "__main__":
    main()
