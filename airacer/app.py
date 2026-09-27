"""Application entrypoint helpers."""

from __future__ import annotations

import argparse


def _load_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            "pygame is required. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    pygame.init()
    pygame.font.init()
    return pygame


def create_game(driver_name: str = "keyboard", policy_path: str | None = None):
    pygame = _load_pygame()

    from .controllers import KeyboardController
    from .game import Game
    from .policies import create_policy
    from .scripted import ScriptedController

    if driver_name == "keyboard":
        controller = KeyboardController(pygame)
    else:
        controller = ScriptedController(create_policy(driver_name, policy_path))

    return Game(pygame, controller=controller, mode_name=driver_name)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run the aiRacer interactive UI.")
    parser.add_argument(
        "--driver",
        default="keyboard",
        choices=["keyboard", "scripted", "linear", "mlp", "neat", "policy"],
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Path to a saved policy file for drivers that require one.",
    )
    args = parser.parse_args(argv)

    game = create_game(driver_name=args.driver, policy_path=args.policy)
    game.run()
