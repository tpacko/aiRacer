"""Core simulation models and deterministic update loop."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from . import config
from .track import (
    DEFAULT_TRACK_NAME,
    Point,
    Track,
    create_track,
    nearest_wall_contact,
    nearest_wall_distance,
    point_in_track,
    progress_on_centerline,
    raycast_walls,
    sample_centerline_ahead,
)


@dataclass
class ControlInput:
    """Normalized control values produced by any controller."""

    steer: float = 0.0
    acceleration: float = 0.0


@dataclass
class SensorReading:
    angle: float
    distance: float
    normalized_distance: float
    hit_point: Point


@dataclass
class LookaheadSample:
    distance: float
    point: Point
    heading_delta: float


@dataclass
class Observation:
    progress: float
    centerline_error: float
    heading_error: float
    speed: float
    forward_speed: float
    lateral_speed: float
    angular_velocity: float
    is_drifting: bool
    collision_count: int
    sensor_readings: list[SensorReading] = field(default_factory=list)
    lookahead_samples: list[LookaheadSample] = field(default_factory=list)
    vector: list[float] | None = None
    min_sensor_normalized: float = 1.0


@dataclass
class CarState:
    """Mutable state for a single simulated car."""

    x: float
    y: float
    heading: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    angular_velocity: float = 0.0

    @property
    def speed(self):
        return math.hypot(self.velocity_x, self.velocity_y)

    @property
    def forward_speed(self):
        return (
            self.velocity_x * math.cos(self.heading)
            + self.velocity_y * math.sin(self.heading)
        )


@dataclass
class SimulationSnapshot:
    car: CarState
    track: Track
    observation: Observation
    crashed: bool = False


class Simulation:
    """Owns world state and advances it in fixed-size steps."""

    def __init__(self, track_name: str = DEFAULT_TRACK_NAME, *, compact_observations: bool = False):
        self.track_name = track_name
        self.compact_observations = compact_observations
        self.track = create_track(track_name)
        self.progress_origin = progress_on_centerline(
            self.track.spawn_position, self.track
        )[0]
        self.steps = 0
        self.crashed = False
        self.collision_count = 0
        self.car = self._spawn_car()
        self.lap_progress = 0.0
        self.max_progress = 0.0
        self.is_drifting = False
        self.observation = self._build_observation()

    def load_track(self, track_name: str, *, randomize_start: bool = False, rng: random.Random | None = None):
        self.track_name = track_name
        self.track = create_track(track_name)
        self.progress_origin = progress_on_centerline(
            self.track.spawn_position, self.track
        )[0]
        self.reset(randomize_start=randomize_start, rng=rng)

    def reset(self, *, randomize_start: bool = False, rng: random.Random | None = None):
        self.steps = 0
        self.crashed = False
        self.collision_count = 0
        if randomize_start:
            self.car = self._spawn_car_random(rng or random.Random())
        else:
            self.progress_origin = progress_on_centerline(
                self.track.spawn_position, self.track
            )[0]
            self.car = self._spawn_car()
        self.lap_progress = 0.0
        self.max_progress = 0.0
        self.is_drifting = False
        self.observation = self._build_observation()
        return self.car

    def step(self, control: ControlInput, dt: float):
        self._advance(control, dt)
        return self.snapshot

    def step_observation(self, control: ControlInput, dt: float):
        self._advance(control, dt)
        return self.observation

    def _advance(self, control: ControlInput, dt: float):
        steer = _clamp(control.steer, -1.0, 1.0)
        acceleration = _clamp(control.acceleration, -1.0, 1.0)

        forward_x = math.cos(self.car.heading)
        forward_y = math.sin(self.car.heading)
        right_x = -forward_y
        right_y = forward_x

        forward_speed = (
            self.car.velocity_x * forward_x + self.car.velocity_y * forward_y
        )

        drifting = (
            abs(steer) > config.CAR_DRIFT_STEER_THRESHOLD
            and forward_speed > config.CAR_DRIFT_MIN_SPEED
        )
        self.is_drifting = drifting

        if acceleration > 0.0:
            speed_ratio = _clamp(forward_speed / config.CAR_MAX_SPEED, 0.0, 1.0)
            power_factor = max(0.0, 1.0 - speed_ratio ** config.CAR_ACCEL_POWER_EXPONENT)
            engine_acceleration = acceleration * config.CAR_ACCELERATION * power_factor
        elif acceleration < 0.0 and forward_speed > 0.0:
            engine_acceleration = acceleration * config.CAR_BRAKE_DECELERATION
        elif acceleration < 0.0:
            engine_acceleration = acceleration * config.CAR_REVERSE_ACCELERATION
        else:
            engine_acceleration = 0.0

        speed_ratio = _clamp(abs(forward_speed) / config.CAR_MAX_SPEED, 0.0, 1.0)
        steer_scale = config.CAR_STEER_AUTHORITY_MIN + (
            1.0 - config.CAR_STEER_AUTHORITY_MIN
        ) * (speed_ratio ** config.CAR_STEER_AUTHORITY_EXPONENT)
        angular_boost = config.CAR_DRIFT_ANGULAR_BOOST if drifting else 1.0
        angular_accel = steer * config.CAR_ANGULAR_ACCEL * steer_scale * angular_boost
        if forward_speed < 0.0:
            angular_accel *= -0.6
        self.car.angular_velocity += angular_accel * dt
        self.car.angular_velocity *= max(0.0, 1.0 - config.CAR_ANGULAR_DRAG * dt)
        self.car.angular_velocity = _clamp(
            self.car.angular_velocity,
            -config.CAR_MAX_ANGULAR_VELOCITY,
            config.CAR_MAX_ANGULAR_VELOCITY,
        )
        self.car.heading += self.car.angular_velocity * dt

        self.car.velocity_x += forward_x * engine_acceleration * dt
        self.car.velocity_y += forward_y * engine_acceleration * dt

        drag_factor = max(0.0, 1.0 - config.CAR_DRAG * dt)
        self.car.velocity_x *= drag_factor
        self.car.velocity_y *= drag_factor

        if acceleration == 0.0:
            rolling_factor = max(0.0, 1.0 - config.CAR_ROLLING_RESISTANCE * dt)
            self.car.velocity_x *= rolling_factor
            self.car.velocity_y *= rolling_factor

        current_velocity_x = self.car.velocity_x
        current_velocity_y = self.car.velocity_y
        forward_speed = current_velocity_x * forward_x + current_velocity_y * forward_y
        lateral_speed = current_velocity_x * right_x + current_velocity_y * right_y
        grip = config.CAR_LATERAL_GRIP * (config.CAR_DRIFT_GRIP_FACTOR if drifting else 1.0)
        lateral_speed *= max(0.0, 1.0 - grip * dt)

        if drifting:
            forward_speed *= max(0.0, 1.0 - config.CAR_DRIFT_FORWARD_DRAG * dt)

        clamped_forward_speed = _clamp(
            forward_speed, config.CAR_MIN_SPEED, config.CAR_MAX_SPEED
        )
        target_velocity_x = forward_x * clamped_forward_speed + right_x * lateral_speed
        target_velocity_y = forward_y * clamped_forward_speed + right_y * lateral_speed
        alignment = config.CAR_TRACTION_ALIGNMENT
        if drifting:
            alignment *= config.CAR_DRIFT_ALIGNMENT_FACTOR
        align_factor = 1.0 - math.exp(-alignment * dt)
        self.car.velocity_x = current_velocity_x + (
            target_velocity_x - current_velocity_x
        ) * align_factor
        self.car.velocity_y = current_velocity_y + (
            target_velocity_y - current_velocity_y
        ) * align_factor

        previous_x = self.car.x
        previous_y = self.car.y
        self.car.x += self.car.velocity_x * dt
        self.car.y += self.car.velocity_y * dt
        self.steps += 1
        self._resolve_wall_collision(previous_x, previous_y)
        self.observation = self._build_observation()

    @property
    def snapshot(self):
        return SimulationSnapshot(
            car=self.car,
            track=self.track,
            observation=self.observation,
            crashed=self.crashed,
        )

    def _spawn_car(self):
        spawn_x, spawn_y = self.track.spawn_position
        return CarState(
            x=spawn_x,
            y=spawn_y,
            heading=self.track.spawn_heading,
        )

    def _spawn_car_random(self, rng: random.Random):
        fallback: CarState | None = None
        fallback_progress = 0.0
        attempts = max(1, int(config.RANDOM_START_MAX_ATTEMPTS))
        for _ in range(attempts):
            car, progress = self._sample_random_spawn_candidate(rng)
            if fallback is None:
                fallback = car
                fallback_progress = progress
            if self._is_drivable_spawn(car):
                self.progress_origin = progress
                return car

        self.progress_origin = fallback_progress
        return fallback if fallback is not None else self._spawn_car()

    def _sample_random_spawn_candidate(self, rng: random.Random):
        progress = rng.random()
        (center_x, center_y), tangent = sample_centerline_ahead(
            self.track, start_progress=progress, arc_distance=0.0
        )
        wall_distance = nearest_wall_distance((center_x, center_y), self.track)
        safe_half = max(0.0, wall_distance - config.CAR_COLLISION_RADIUS - 4.0)
        lateral_max = min(
            config.RANDOM_START_LATERAL_JITTER,
            safe_half * config.RANDOM_START_LATERAL_FRACTION,
        )
        lateral_offset = rng.uniform(-lateral_max, lateral_max) if lateral_max > 0.0 else 0.0
        normal_x = -math.sin(tangent)
        normal_y = math.cos(tangent)
        spawn_x = center_x + normal_x * lateral_offset
        spawn_y = center_y + normal_y * lateral_offset
        if not point_in_track((spawn_x, spawn_y), self.track):
            spawn_x, spawn_y = center_x, center_y
        heading = tangent + rng.uniform(
            -config.RANDOM_START_HEADING_JITTER, config.RANDOM_START_HEADING_JITTER
        )
        progress = progress_on_centerline((spawn_x, spawn_y), self.track)[0]
        forward_clearance = self._forward_clearance((spawn_x, spawn_y), heading)
        speed_max = min(
            config.RANDOM_START_FORWARD_SPEED_MAX,
            max(
                config.RANDOM_START_FORWARD_SPEED_MIN,
                (forward_clearance - config.RANDOM_START_MIN_FORWARD_CLEARANCE) * 2.0,
            ),
        )
        speed = rng.uniform(config.RANDOM_START_FORWARD_SPEED_MIN, speed_max)
        return (
            CarState(
                x=spawn_x,
                y=spawn_y,
                heading=heading,
                velocity_x=math.cos(heading) * speed,
                velocity_y=math.sin(heading) * speed,
            ),
            progress,
        )

    def _forward_clearance(self, point: Point, heading: float):
        _, distance, _ = raycast_walls(
            point, heading, config.SENSOR_RAY_LENGTH, self.track
        )
        return distance

    def _is_drivable_spawn(self, car: CarState):
        point = (car.x, car.y)
        if not point_in_track(point, self.track):
            return False
        if nearest_wall_distance(point, self.track) <= config.CAR_COLLISION_RADIUS + 2.0:
            return False
        return (
            self._forward_clearance(point, car.heading)
            >= config.RANDOM_START_MIN_FORWARD_CLEARANCE
        )

    def _resolve_wall_collision(self, previous_x: float, previous_y: float):
        point = (self.car.x, self.car.y)
        wall_distance, segment, closest_point = nearest_wall_contact(point, self.track)
        inside_track = point_in_track(point, self.track)
        if inside_track and wall_distance > config.CAR_COLLISION_RADIUS:
            return

        self.collision_count += 1

        tangent_x = segment[1][0] - segment[0][0]
        tangent_y = segment[1][1] - segment[0][1]
        tangent_length = math.hypot(tangent_x, tangent_y) or 1.0
        tangent_x /= tangent_length
        tangent_y /= tangent_length

        normal_candidates = (
            (-tangent_y, tangent_x),
            (tangent_y, -tangent_x),
        )
        push_normal = self._pick_track_facing_normal(closest_point, normal_candidates)
        if push_normal is None:
            self.car.x = previous_x
            self.car.y = previous_y
            self.car.velocity_x *= config.CAR_COLLISION_SPEED_LOSS
            self.car.velocity_y *= config.CAR_COLLISION_SPEED_LOSS
            return

        push_distance = config.CAR_COLLISION_RADIUS + 2.0
        self.car.x = closest_point[0] + push_normal[0] * push_distance
        self.car.y = closest_point[1] + push_normal[1] * push_distance

        if not point_in_track((self.car.x, self.car.y), self.track):
            self.car.x = previous_x
            self.car.y = previous_y

        velocity_normal = (
            self.car.velocity_x * push_normal[0] + self.car.velocity_y * push_normal[1]
        )
        if velocity_normal < 0.0:
            self.car.velocity_x -= (
                (1.0 + config.CAR_COLLISION_BOUNCE)
                * velocity_normal
                * push_normal[0]
            )
            self.car.velocity_y -= (
                (1.0 + config.CAR_COLLISION_BOUNCE)
                * velocity_normal
                * push_normal[1]
            )

        self.car.velocity_x *= config.CAR_COLLISION_SPEED_LOSS
        self.car.velocity_y *= config.CAR_COLLISION_SPEED_LOSS
        self.car.angular_velocity *= config.CAR_COLLISION_ANGULAR_LOSS

    def _build_observation(self):
        point = (self.car.x, self.car.y)
        raw_progress, centerline_error = progress_on_centerline(point, self.track)
        progress = (raw_progress - self.progress_origin) % 1.0
        self.lap_progress = progress
        self.max_progress = max(self.max_progress, progress)
        path_heading = self._path_heading(raw_progress)
        heading_error = _normalize_angle(self.car.heading - path_heading)
        forward_x = math.cos(self.car.heading)
        forward_y = math.sin(self.car.heading)
        speed = self.car.speed
        forward_speed = self.car.forward_speed
        lateral_speed = -self.car.velocity_x * forward_y + self.car.velocity_y * forward_x
        if self.compact_observations:
            sensor_values, min_sensor_normalized = self._sensor_distances(point)
            lookahead_values = self._lookahead_heading_deltas(raw_progress, path_heading)
            angular_velocity_norm = self.car.angular_velocity / max(
                config.CAR_MAX_ANGULAR_VELOCITY, 1e-6
            )
            vector = [
                centerline_error / config.SENSOR_RAY_LENGTH,
                heading_error / math.pi,
                speed / config.CAR_MAX_SPEED,
                forward_speed / config.CAR_MAX_SPEED,
                *sensor_values,
                lateral_speed / config.CAR_MAX_SPEED,
                angular_velocity_norm,
                *lookahead_values,
            ]
            return Observation(
                progress=progress,
                centerline_error=centerline_error,
                heading_error=heading_error,
                speed=speed,
                forward_speed=forward_speed,
                lateral_speed=lateral_speed,
                angular_velocity=self.car.angular_velocity,
                is_drifting=self.is_drifting,
                collision_count=self.collision_count,
                vector=vector,
                min_sensor_normalized=min_sensor_normalized,
            )

        sensor_readings = self._sensor_readings(point)
        lookahead_samples = self._lookahead_samples(raw_progress, path_heading)
        return Observation(
            progress=progress,
            centerline_error=centerline_error,
            heading_error=heading_error,
            speed=speed,
            forward_speed=forward_speed,
            lateral_speed=lateral_speed,
            angular_velocity=self.car.angular_velocity,
            is_drifting=self.is_drifting,
            collision_count=self.collision_count,
            sensor_readings=sensor_readings,
            lookahead_samples=lookahead_samples,
            min_sensor_normalized=min(
                (reading.normalized_distance for reading in sensor_readings),
                default=1.0,
            ),
        )

    def _lookahead_samples(self, raw_progress: float, current_path_heading: float):
        samples: list[LookaheadSample] = []
        for distance in config.LOOKAHEAD_DISTANCES:
            point, tangent = sample_centerline_ahead(
                self.track, start_progress=raw_progress, arc_distance=distance
            )
            heading_delta = _normalize_angle(tangent - current_path_heading)
            samples.append(
                LookaheadSample(
                    distance=distance,
                    point=point,
                    heading_delta=heading_delta,
                )
            )
        return samples

    def _lookahead_heading_deltas(self, raw_progress: float, current_path_heading: float):
        values: list[float] = []
        for distance in config.LOOKAHEAD_DISTANCES:
            _, tangent = sample_centerline_ahead(
                self.track, start_progress=raw_progress, arc_distance=distance
            )
            values.append(_normalize_angle(tangent - current_path_heading) / math.pi)
        return values

    def _pick_track_facing_normal(self, closest_point: Point, normals):
        for normal_x, normal_y in normals:
            probe_point = (
                closest_point[0] + normal_x * (config.CAR_COLLISION_RADIUS + 4.0),
                closest_point[1] + normal_y * (config.CAR_COLLISION_RADIUS + 4.0),
            )
            if point_in_track(probe_point, self.track):
                return normal_x, normal_y
        return None

    def _sensor_readings(self, point: Point):
        readings = []
        for relative_angle in config.SENSOR_RAY_ANGLES:
            ray_angle = self.car.heading + relative_angle
            _, distance, hit_point = raycast_walls(
                point,
                ray_angle,
                config.SENSOR_RAY_LENGTH,
                self.track,
            )
            readings.append(
                SensorReading(
                    angle=relative_angle,
                    distance=distance,
                    normalized_distance=distance / config.SENSOR_RAY_LENGTH,
                    hit_point=hit_point,
                )
            )
        return readings

    def _sensor_distances(self, point: Point):
        values: list[float] = []
        min_value = 1.0
        for relative_angle in config.SENSOR_RAY_ANGLES:
            ray_angle = self.car.heading + relative_angle
            _, distance, _ = raycast_walls(
                point,
                ray_angle,
                config.SENSOR_RAY_LENGTH,
                self.track,
            )
            normalized = distance / config.SENSOR_RAY_LENGTH
            values.append(normalized)
            if normalized < min_value:
                min_value = normalized
        return values, min_value

    def _path_heading(self, progress: float):
        segments = self.track.centerline_segments
        segment_lengths = self.track.centerline_segment_lengths
        total_length = self.track.centerline_total_length

        target = progress * total_length
        traversed = 0.0
        for index, segment in enumerate(segments):
            length = segment_lengths[index]
            if traversed + length >= target:
                start, end = segment
                return math.atan2(end[1] - start[1], end[0] - start[0])
            traversed += length

        start, end = segments[-1]
        return math.atan2(end[1] - start[1], end[0] - start[0])


def _clamp(value: float, minimum: float, maximum: float):
    return max(minimum, min(maximum, value))


def _normalize_angle(angle: float):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
