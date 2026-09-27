"""Controller interfaces and manual input implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .simulation import ControlInput, Simulation


@dataclass
class ControllerCommand:
    """High-level command output from a controller."""

    control: ControlInput
    reset_requested: bool = False


class Controller(Protocol):
    """Common controller interface for keyboard and future AI drivers."""

    def sample(self, simulation: Simulation) -> ControllerCommand:
        ...


class KeyboardController:
    """Maps keyboard state to normalized car controls."""

    def __init__(self, pygame_module):
        self.pygame = pygame_module

    def sample(self, simulation: Simulation) -> ControllerCommand:
        keys = self.pygame.key.get_pressed()

        acceleration = 0.0
        if keys[self.pygame.K_UP] or keys[self.pygame.K_w]:
            acceleration += 1.0
        if keys[self.pygame.K_DOWN] or keys[self.pygame.K_s]:
            acceleration -= 1.0

        steer = 0.0
        if keys[self.pygame.K_LEFT] or keys[self.pygame.K_a]:
            steer -= 1.0
        if keys[self.pygame.K_RIGHT] or keys[self.pygame.K_d]:
            steer += 1.0

        reset_requested = bool(keys[self.pygame.K_r])

        return ControllerCommand(
            control=ControlInput(steer=steer, acceleration=acceleration),
            reset_requested=reset_requested,
        )
