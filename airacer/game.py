"""Main game loop and rendering for the initial scaffold."""

from __future__ import annotations

import math
import time

from . import config
from .controllers import KeyboardController
from .env import RaceEnv
from .simulation import ControlInput
from .track import (
    DEFAULT_TRACK_NAME,
    REVERSE_SUFFIX,
    _base_track_name,
    available_track_names,
    track_bounding_box,
)


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return "%d:%05.2f" % (minutes, remainder)


CAMERA_MARGIN = 40.0


class Game:
    """Owns the application lifecycle and fixed-timestep loop."""

    def __init__(self, pygame_module, controller=None, mode_name="keyboard"):
        self.pygame = pygame_module
        self.screen = self.pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        self.pygame.display.set_caption("%s - %s" % (config.WINDOW_TITLE, mode_name))
        self.clock = self.pygame.time.Clock()
        self.font_large = self.pygame.font.SysFont("consolas", 28, bold=True)
        self.font = self.pygame.font.SysFont("consolas", 18)
        self.font_small = self.pygame.font.SysFont("consolas", 14)
        self.running = True
        self.track_names = available_track_names()
        self.track_index = (
            self.track_names.index(DEFAULT_TRACK_NAME)
            if DEFAULT_TRACK_NAME in self.track_names
            else 0
        )
        self.env = RaceEnv(max_steps=2500, track_name=self.track_names[self.track_index])
        self.env.reset()
        self.simulation = self.env.simulation
        self.simulation_tick = self.env.dt
        self.accumulator = 0.0
        self.previous_time = time.perf_counter()
        self.frame_count = 0
        self.controller = controller or KeyboardController(self.pygame)
        self.mode_name = mode_name
        self.control_input = ControlInput()
        self.reset_latch = False
        self.switch_latch = False
        self.lap_time = 0.0
        self.best_lap_time = None
        self.best_lap_times = {name: None for name in self.track_names}
        self.previous_progress = self.simulation.observation.progress
        self._camera_scale = 1.0
        self._camera_offset = (0.0, 0.0)
        self._recompute_camera()

    def run(self):
        while self.running:
            current_time = time.perf_counter()
            frame_time = min(current_time - self.previous_time, 0.25)
            self.previous_time = current_time
            self.accumulator += frame_time

            self._pump_events()

            while self.accumulator >= self.simulation_tick:
                self._sample_controller()
                self._simulate(self.simulation_tick)
                self.accumulator -= self.simulation_tick

            self._render()
            self.clock.tick(config.RENDER_HZ)

        self.pygame.quit()

    def _pump_events(self):
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.running = False
            elif event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_ESCAPE:
                    self.running = False
                elif event.key == self.pygame.K_t:
                    mods = self.pygame.key.get_mods()
                    direction = -1 if mods & self.pygame.KMOD_SHIFT else 1
                    self._cycle_track(direction=direction)
                elif event.key == self.pygame.K_f:
                    self._flip_current_track()

    def _sample_controller(self):
        command = self.controller.sample(self.simulation)
        self.control_input = command.control
        if command.reset_requested and not self.reset_latch:
            self.env.reset()
            self.lap_time = 0.0
            self.previous_progress = self.simulation.observation.progress
            self.reset_latch = True
        elif not command.reset_requested:
            self.reset_latch = False

    def _cycle_track(self, direction: int = 1):
        self.track_index = (self.track_index + direction) % len(self.track_names)
        self._load_track_at_index(self.track_index)

    def _flip_current_track(self):
        """Switch between the forward and ``_rev`` variant of the current
        track — walls unchanged, traversal direction mirrored."""
        current = self.track_names[self.track_index]
        base, is_reverse = _base_track_name(current)
        target = base if is_reverse else base + REVERSE_SUFFIX
        if target in self.track_names:
            self.track_index = self.track_names.index(target)
            self._load_track_at_index(self.track_index)

    def _load_track_at_index(self, index: int):
        name = self.track_names[index]
        self.env.load_track(name)
        self.simulation = self.env.simulation
        self.lap_time = 0.0
        self.previous_progress = self.simulation.observation.progress
        self._recompute_camera()
        self.pygame.display.set_caption(
            "%s - %s [%s]" % (config.WINDOW_TITLE, self.mode_name, name)
        )

    def _recompute_camera(self):
        min_x, min_y, max_x, max_y = track_bounding_box(self.simulation.track)
        width = max(max_x - min_x, 1.0)
        height = max(max_y - min_y, 1.0)
        available_w = config.WINDOW_WIDTH - 2.0 * CAMERA_MARGIN
        available_h = config.WINDOW_HEIGHT - 2.0 * CAMERA_MARGIN
        scale = min(available_w / width, available_h / height)
        self._camera_scale = scale
        scaled_w = width * scale
        scaled_h = height * scale
        offset_x = CAMERA_MARGIN + (available_w - scaled_w) * 0.5 - min_x * scale
        offset_y = CAMERA_MARGIN + (available_h - scaled_h) * 0.5 - min_y * scale
        self._camera_offset = (offset_x, offset_y)

    def _w2s(self, point):
        ox, oy = self._camera_offset
        return (
            int(point[0] * self._camera_scale + ox),
            int(point[1] * self._camera_scale + oy),
        )

    def _w2s_f(self, point):
        ox, oy = self._camera_offset
        return (
            point[0] * self._camera_scale + ox,
            point[1] * self._camera_scale + oy,
        )

    def _scale(self, value):
        return value * self._camera_scale

    def _simulate(self, dt):
        self.env.step((self.control_input.steer, self.control_input.acceleration))
        self.lap_time += dt
        current_progress = self.simulation.observation.progress
        if (
            self.previous_progress > 0.85
            and current_progress < 0.15
            and self.simulation.max_progress > 0.9
        ):
            if self.best_lap_time is None or self.lap_time < self.best_lap_time:
                self.best_lap_time = self.lap_time
            current_name = self.track_names[self.track_index]
            prior_best = self.best_lap_times[current_name]
            if prior_best is None or self.lap_time < prior_best:
                self.best_lap_times[current_name] = self.lap_time
            self.lap_time = 0.0
        self.previous_progress = current_progress

    def _render(self):
        self.screen.fill(config.TRACK_GRASS_COLOR)
        self._draw_track()
        self._draw_car()
        self._draw_status()
        self._draw_track_list()
        self.pygame.display.flip()
        self.frame_count += 1

    def _draw_status(self):
        observation = self.simulation.observation
        speed_pct = observation.speed / config.CAR_MAX_SPEED

        speed_surface = self.font_large.render(
            "%3d" % int(observation.speed), True, config.TEXT_COLOR
        )
        self.screen.blit(speed_surface, (24, 20))
        self.screen.blit(
            self.font_small.render("px/s", True, config.TEXT_COLOR),
            (24 + speed_surface.get_width() + 6, 36),
        )
        self._draw_bar(24, 56, 120, 6, speed_pct, config.CAR_NOSE_COLOR)

        lap_text = "LAP  %s" % _format_time(self.lap_time)
        self.screen.blit(self.font.render(lap_text, True, config.TEXT_COLOR), (24, 78))
        current_name = self.track_names[self.track_index]
        best_for_track = self.best_lap_times[current_name]
        if best_for_track is not None:
            best_text = "BEST %s" % _format_time(best_for_track)
            self.screen.blit(self.font_small.render(best_text, True, config.TEXT_COLOR), (24, 100))

        progress_text = "%4.1f%%" % (observation.progress * 100.0)
        self.screen.blit(self.font.render(progress_text, True, config.TEXT_COLOR), (24, 120))
        self._draw_bar(24, 142, 120, 4, observation.progress, config.PROGRESS_COLOR)

        # Step counter — shows how many simulation steps this episode has
        # consumed. Useful for correlating driver behaviour with training-step
        # budgets ("can it still make the hairpin at step 1800?").
        steps_text = "STEPS %d" % self.env.step_count
        self.screen.blit(
            self.font_small.render(steps_text, True, config.TEXT_COLOR),
            (24, 154),
        )

        if observation.collision_count > 0:
            hits = self.font_small.render(
                "HITS %d" % observation.collision_count, True, (240, 120, 120)
            )
            self.screen.blit(hits, (24, 170))

        if observation.is_drifting:
            drift = self.font.render("DRIFT", True, config.DRIFT_INDICATOR_COLOR)
            self.screen.blit(drift, (24, config.WINDOW_HEIGHT - 34))

        self._draw_control_bars()
        self._draw_policy_info()

        fps_text = self.font_small.render("%d fps" % int(self.clock.get_fps()), True, config.TEXT_COLOR)
        self.screen.blit(fps_text, (config.WINDOW_WIDTH - fps_text.get_width() - 12, 12))

    def _draw_policy_info(self):
        """If a policy with attached metadata is driving, show how many
        training steps produced it and which curriculum stage it came from."""
        metadata = getattr(self.controller, "policy_metadata", None)
        if not metadata:
            return
        lines = ["POLICY"]
        trained_steps = metadata.get("trained_steps")
        if trained_steps is not None:
            lines.append("trained_steps %s" % f"{int(trained_steps):,}")
        stage_name = metadata.get("stage_name")
        stage_index = metadata.get("stage_index")
        total_stages = metadata.get("total_stages")
        if stage_name is not None:
            if stage_index is not None and total_stages is not None:
                lines.append(
                    "stage %d/%d %s"
                    % (int(stage_index) + 1, int(total_stages), stage_name)
                )
            else:
                lines.append("stage %s" % stage_name)
        best_return = metadata.get("best_mean_return")
        if best_return is not None:
            lines.append("best_mean_return %+0.1f" % float(best_return))
        snapshot = metadata.get("snapshot")
        if snapshot:
            lines.append("snapshot %s" % snapshot)

        x = 160
        y = 20
        for index, text in enumerate(lines):
            color = config.TEXT_COLOR if index > 0 else config.PROGRESS_COLOR
            surface = self.font_small.render(text, True, color)
            self.screen.blit(surface, (x, y + index * 16))

    def _draw_track_list(self):
        # With basic/intermediate/advanced tracks and each of their reversed
        # variants we can have 14+ entries, so the panel is a touch wider
        # and we split the list into two columns so nothing runs off-screen.
        panel_width = 260
        panel_x = config.WINDOW_WIDTH - panel_width - 12
        panel_y = 40
        header = self.font_small.render(
            "TRACKS  [T]/[shift+T] cycle  [F] flip", True, config.TEXT_COLOR
        )
        self.screen.blit(header, (panel_x, panel_y))

        total = len(self.track_names)
        per_column = (total + 1) // 2
        col_width = panel_width // 2
        row_height = 16
        for index, name in enumerate(self.track_names):
            col = index // per_column
            row = index % per_column
            marker = ">" if index == self.track_index else " "
            text = "%s %s" % (marker, name)
            color = (
                config.PROGRESS_COLOR if index == self.track_index else config.TEXT_COLOR
            )
            surface = self.font_small.render(text, True, color)
            self.screen.blit(
                surface,
                (panel_x + col * col_width, panel_y + 20 + row * row_height),
            )

    def _draw_bar(self, x, y, width, height, value, color):
        value = max(0.0, min(1.0, value))
        self.pygame.draw.rect(self.screen, (40, 46, 54), (x, y, width, height))
        self.pygame.draw.rect(self.screen, color, (x, y, int(width * value), height))

    def _draw_bipolar_bar(self, x, y, width, height, value, positive_color, negative_color):
        value = max(-1.0, min(1.0, value))
        self.pygame.draw.rect(self.screen, (40, 46, 54), (x, y, width, height))
        midpoint = x + width // 2
        self.pygame.draw.line(self.screen, (92, 100, 110), (midpoint, y), (midpoint, y + height), 1)
        if value > 0.0:
            fill = int((width // 2) * value)
            self.pygame.draw.rect(self.screen, positive_color, (midpoint, y, fill, height))
        elif value < 0.0:
            fill = int((width // 2) * -value)
            self.pygame.draw.rect(self.screen, negative_color, (midpoint - fill, y, fill, height))

    def _draw_control_bars(self):
        base_x = 24
        base_y = config.WINDOW_HEIGHT - 54
        self.screen.blit(self.font_small.render("ACC", True, config.TEXT_COLOR), (base_x, base_y))
        self._draw_bipolar_bar(
            base_x + 34,
            base_y + 4,
            90,
            6,
            self.control_input.acceleration,
            (120, 220, 140),
            (220, 120, 120),
        )
        self.screen.blit(self.font_small.render("STR", True, config.TEXT_COLOR), (base_x, base_y + 16))
        self._draw_bipolar_bar(
            base_x + 34,
            base_y + 20,
            90,
            6,
            self.control_input.steer,
            (140, 180, 230),
            (140, 180, 230),
        )

    def _draw_car(self):
        car = self.simulation.car
        heading = car.heading
        forward_x = math.cos(heading)
        forward_y = math.sin(heading)
        right_x = -forward_y
        right_y = forward_x

        scale = self._camera_scale
        half_length = config.CAR_LENGTH * 0.5
        half_width = config.CAR_WIDTH * 0.5

        def world(dx, dy):
            return (car.x + dx, car.y + dy)

        nose = world(forward_x * half_length, forward_y * half_length)
        front_left = world(
            forward_x * (half_length * 0.65) - right_x * half_width,
            forward_y * (half_length * 0.65) - right_y * half_width,
        )
        front_right = world(
            forward_x * (half_length * 0.65) + right_x * half_width,
            forward_y * (half_length * 0.65) + right_y * half_width,
        )
        rear_left = world(
            -forward_x * half_length - right_x * half_width,
            -forward_y * half_length - right_y * half_width,
        )
        rear_right = world(
            -forward_x * half_length + right_x * half_width,
            -forward_y * half_length + right_y * half_width,
        )

        self.pygame.draw.polygon(
            self.screen,
            config.CAR_COLOR,
            [self._w2s(nose), self._w2s(front_right), self._w2s(rear_right), self._w2s(rear_left), self._w2s(front_left)],
        )
        cabin_center = world(-forward_x * 2.0, -forward_y * 2.0)
        cabin_half_length = config.CAR_LENGTH * 0.22
        cabin_half_width = config.CAR_WIDTH * 0.28
        windshield_points = [
            (
                cabin_center[0] + forward_x * cabin_half_length - right_x * cabin_half_width,
                cabin_center[1] + forward_y * cabin_half_length - right_y * cabin_half_width,
            ),
            (
                cabin_center[0] + forward_x * cabin_half_length + right_x * cabin_half_width,
                cabin_center[1] + forward_y * cabin_half_length + right_y * cabin_half_width,
            ),
            (
                cabin_center[0] - forward_x * cabin_half_length + right_x * cabin_half_width,
                cabin_center[1] - forward_y * cabin_half_length + right_y * cabin_half_width,
            ),
            (
                cabin_center[0] - forward_x * cabin_half_length - right_x * cabin_half_width,
                cabin_center[1] - forward_y * cabin_half_length - right_y * cabin_half_width,
            ),
        ]
        self.pygame.draw.polygon(
            self.screen,
            config.CAR_WINDOW_COLOR,
            [self._w2s(p) for p in windshield_points],
        )
        wheel_w = max(3, int(6 * scale))
        wheel_h = max(5, int(12 * scale))
        for wheel_center in self._wheel_centers(car.x, car.y, forward_x, forward_y, right_x, right_y):
            wheel_rect = self.pygame.Rect(0, 0, wheel_w, wheel_h)
            wheel_rect.center = self._w2s(wheel_center)
            self.pygame.draw.rect(self.screen, config.CAR_WHEEL_COLOR, wheel_rect, border_radius=2)
        self.pygame.draw.circle(
            self.screen,
            config.CAR_NOSE_COLOR,
            self._w2s(nose),
            max(2, int(4 * scale)),
        )

    def _wheel_centers(self, x, y, forward_x, forward_y, right_x, right_y):
        front_offset = config.CAR_LENGTH * 0.22
        rear_offset = config.CAR_LENGTH * 0.22
        side_offset = config.CAR_WIDTH * 0.38
        return [
            (
                x + forward_x * front_offset - right_x * side_offset,
                y + forward_y * front_offset - right_y * side_offset,
            ),
            (
                x + forward_x * front_offset + right_x * side_offset,
                y + forward_y * front_offset + right_y * side_offset,
            ),
            (
                x - forward_x * rear_offset - right_x * side_offset,
                y - forward_y * rear_offset - right_y * side_offset,
            ),
            (
                x - forward_x * rear_offset + right_x * side_offset,
                y - forward_y * rear_offset + right_y * side_offset,
            ),
        ]

    def _draw_track(self):
        track = self.simulation.track
        outer = [self._w2s(p) for p in track.outer_points]
        inner = [self._w2s(p) for p in track.inner_points]
        centerline = [self._w2s(p) for p in track.centerline_points]
        self.pygame.draw.polygon(self.screen, config.TRACK_SURFACE_COLOR, outer)
        self.pygame.draw.polygon(self.screen, config.TRACK_INFIELD_COLOR, inner)
        self.pygame.draw.lines(
            self.screen,
            config.TRACK_CENTERLINE_COLOR,
            False,
            centerline,
            max(1, int(2 * self._camera_scale)),
        )
        shoulder_w = max(2, int(4 * self._camera_scale))
        self.pygame.draw.lines(self.screen, config.TRACK_SHOULDER_COLOR, True, outer, shoulder_w)
        self.pygame.draw.lines(self.screen, config.TRACK_SHOULDER_COLOR, True, inner, shoulder_w)
        spawn_x, spawn_y = track.spawn_position
        marker_world = (spawn_x - 20, spawn_y)
        marker_center = self._w2s(marker_world)
        marker_rect = self.pygame.Rect(
            0, 0, max(4, int(12 * self._camera_scale)), max(12, int(40 * self._camera_scale))
        )
        marker_rect.center = marker_center
        self.pygame.draw.rect(self.screen, (244, 244, 244), marker_rect)
        self.pygame.draw.rect(self.screen, (20, 20, 20), marker_rect, width=2)
        self._draw_progress_marker()
        self._draw_lookahead()
        self._draw_sensor_rays()

    def _draw_sensor_rays(self):
        car = self.simulation.car
        readings = self.simulation.observation.sensor_readings
        origin = self._w2s((car.x, car.y))
        for reading in readings:
            hit_point = self._w2s(reading.hit_point)
            self.pygame.draw.line(self.screen, config.SENSOR_RAY_COLOR, origin, hit_point, 1)
            self.pygame.draw.circle(
                self.screen, config.SENSOR_HIT_COLOR, hit_point, max(2, int(3 * self._camera_scale))
            )

    def _draw_lookahead(self):
        samples = self.simulation.observation.lookahead_samples
        if not samples:
            return
        car = self.simulation.car
        origin = self._w2s((car.x, car.y))
        prev = origin
        for sample in samples:
            current = self._w2s(sample.point)
            self.pygame.draw.line(
                self.screen, config.LOOKAHEAD_LINE_COLOR, prev, current, 1
            )
            self.pygame.draw.circle(
                self.screen,
                config.LOOKAHEAD_POINT_COLOR,
                current,
                max(2, int(4 * self._camera_scale)),
            )
            prev = current

    def _draw_progress_marker(self):
        target_point = self._point_on_centerline(self.simulation.observation.progress)
        self.pygame.draw.circle(
            self.screen,
            config.PROGRESS_COLOR,
            self._w2s(target_point),
            max(3, int(6 * self._camera_scale)),
        )

    def _point_on_centerline(self, progress):
        segments = self.simulation.track.centerline_segments
        lengths = []
        total_length = 0.0
        for start, end in segments:
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            lengths.append(length)
            total_length += length

        target = progress * total_length
        traversed = 0.0
        for index, segment in enumerate(segments):
            length = lengths[index]
            if traversed + length >= target:
                start, end = segment
                t = 0.0 if length == 0.0 else (target - traversed) / length
                return (
                    start[0] + (end[0] - start[0]) * t,
                    start[1] + (end[1] - start[1]) * t,
                )
            traversed += length
        return self.simulation.track.centerline_points[-1]
