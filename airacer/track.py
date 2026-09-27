"""Track geometry, spawn points, and collision helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


Point = Tuple[float, float]
Segment = Tuple[Point, Point]
GridCell = Tuple[int, int]


WALL_GRID_CELL_SIZE = 64.0


@dataclass(frozen=True)
class WallSpatialIndex:
    cell_size: float
    cells: Dict[GridCell, Tuple[int, ...]]
    segment_bounds: Tuple[Tuple[float, float, float, float], ...]
    min_cell_x: int
    max_cell_x: int
    min_cell_y: int
    max_cell_y: int


@dataclass(frozen=True)
class Track:
    outer_points: Sequence[Point]
    inner_points: Sequence[Point]
    centerline_points: Sequence[Point]
    spawn_position: Point
    spawn_heading: float
    wall_segments: Tuple[Segment, ...]
    centerline_segments: Tuple[Segment, ...]
    centerline_segment_lengths: Tuple[float, ...]
    centerline_total_length: float
    wall_index: WallSpatialIndex


def _build_track_from_control_points(
    control_points: Sequence[Point],
    width: float = 132.0,
    smoothing_iterations: int = 3,
) -> Track:
    loop_points = _chaikin_closed(list(control_points), iterations=smoothing_iterations)
    centerline_loop = _rotate_loop_start_near_point(loop_points, control_points[0])
    spawn_position = centerline_loop[0]
    spawn_heading = math.atan2(
        centerline_loop[1][1] - centerline_loop[0][1],
        centerline_loop[1][0] - centerline_loop[0][0],
    )
    half_width = width * 0.5
    offset_a = tuple(
        _remove_self_intersections(_offset_closed_loop(centerline_loop, offset=half_width))
    )
    offset_b = tuple(
        _remove_self_intersections(_offset_closed_loop(centerline_loop, offset=-half_width))
    )
    if _point_in_polygon(spawn_position, offset_a) and not _point_in_polygon(
        spawn_position, offset_b
    ):
        outer_points = offset_a
        inner_points = offset_b
    else:
        outer_points = offset_b
        inner_points = offset_a
    centerline_points = tuple(_closed_polyline(centerline_loop))
    wall_segments = tuple(
        _polygon_segments(outer_points) + _polygon_segments(inner_points)
    )
    centerline_segments = tuple(_polyline_segments(centerline_points))
    centerline_segment_lengths = tuple(
        segment_length_between(start, end) for start, end in centerline_segments
    )
    centerline_total_length = sum(centerline_segment_lengths)
    wall_index = _build_wall_spatial_index(wall_segments)
    return Track(
        outer_points=outer_points,
        inner_points=inner_points,
        centerline_points=centerline_points,
        spawn_position=spawn_position,
        spawn_heading=spawn_heading,
        wall_segments=wall_segments,
        centerline_segments=centerline_segments,
        centerline_segment_lengths=centerline_segment_lengths,
        centerline_total_length=centerline_total_length,
        wall_index=wall_index,
    )


def track_bounding_box(track: Track) -> Tuple[float, float, float, float]:
    xs = [x for x, _ in track.outer_points]
    ys = [y for _, y in track.outer_points]
    return min(xs), min(ys), max(xs), max(ys)


def create_basic_oval_track() -> Track:
    """Basic curriculum track #1: wide smooth oval. All turns are the same
    direction — easiest possible "go round the loop" shape. Wide enough that
    even a rough policy can stay on track.
    """
    control_points = [
        (940.0, 360.0),
        (900.0, 490.0),
        (780.0, 550.0),
        (640.0, 570.0),
        (500.0, 550.0),
        (380.0, 490.0),
        (340.0, 360.0),
        (380.0, 230.0),
        (500.0, 170.0),
        (640.0, 150.0),
        (780.0, 170.0),
        (900.0, 230.0),
    ]
    return _build_track_from_control_points(control_points, width=160.0)


def create_basic_rounded_track() -> Track:
    """Basic curriculum track #2: rounded rectangle with longer straights.
    Slightly narrower than the oval, introduces sustained straight segments
    and corner exits."""
    control_points = [
        (1020.0, 380.0),
        (1000.0, 510.0),
        (900.0, 590.0),
        (760.0, 610.0),
        (520.0, 610.0),
        (380.0, 590.0),
        (280.0, 510.0),
        (260.0, 380.0),
        (260.0, 280.0),
        (340.0, 190.0),
        (500.0, 160.0),
        (780.0, 160.0),
        (940.0, 190.0),
        (1020.0, 280.0),
    ]
    return _build_track_from_control_points(control_points, width=150.0)


def create_intermediate_snake_track() -> Track:
    """Intermediate curriculum track #1: snake-like loop with S-curves that
    force alternating left/right steering within a single lap."""
    control_points = [
        (220.0, 360.0),
        (260.0, 220.0),
        (400.0, 180.0),
        (520.0, 260.0),
        (620.0, 200.0),
        (760.0, 180.0),
        (880.0, 260.0),
        (1000.0, 200.0),
        (1100.0, 340.0),
        (1050.0, 500.0),
        (920.0, 560.0),
        (800.0, 520.0),
        (680.0, 580.0),
        (560.0, 540.0),
        (440.0, 580.0),
        (320.0, 530.0),
        (220.0, 470.0),
    ]
    return _build_track_from_control_points(control_points, width=110.0)


def create_intermediate_hairpin_track() -> Track:
    """Intermediate curriculum track #2: wide flowing loop with one tight
    hairpin corner that rewards braking before the apex."""
    control_points = [
        (220.0, 400.0),
        (240.0, 250.0),
        (360.0, 170.0),
        (520.0, 160.0),
        (680.0, 180.0),
        (820.0, 170.0),
        (960.0, 200.0),
        (1080.0, 300.0),
        (1080.0, 440.0),
        (980.0, 520.0),
        (840.0, 520.0),
        (740.0, 450.0),
        (760.0, 360.0),
        (640.0, 320.0),
        (520.0, 380.0),
        (500.0, 500.0),
        (580.0, 590.0),
        (420.0, 620.0),
        (270.0, 580.0),
    ]
    return _build_track_from_control_points(control_points, width=110.0)


def create_test_track() -> Track:
    control_points = [
        (190.0, 270.0),
        (255.0, 185.0),
        (370.0, 145.0),
        (520.0, 120.0),
        (690.0, 135.0),
        (845.0, 118.0),
        (975.0, 165.0),
        (1075.0, 255.0),
        (1100.0, 360.0),
        (1050.0, 445.0),
        (965.0, 500.0),
        (885.0, 555.0),
        (930.0, 630.0),
        (850.0, 665.0),
        (710.0, 640.0),
        (610.0, 585.0),
        (520.0, 640.0),
        (400.0, 655.0),
        (295.0, 615.0),
        (225.0, 540.0),
        (205.0, 450.0),
        (245.0, 380.0),
    ]
    return _build_track_from_control_points(control_points, width=132.0)


def create_technical_sprint_track() -> Track:
    """Short, tight circuit: hairpin, chicane, and opposite-rotation S-curves.

    Narrower than the basic loop — penalises sloppy lines and rewards accurate
    braking. Fits inside the default window.
    """
    control_points = [
        (220.0, 360.0),
        (260.0, 230.0),
        (380.0, 175.0),
        (520.0, 195.0),
        (600.0, 290.0),
        (560.0, 380.0),
        (455.0, 410.0),
        (430.0, 500.0),
        (540.0, 560.0),
        (690.0, 545.0),
        (760.0, 440.0),
        (720.0, 320.0),
        (790.0, 200.0),
        (930.0, 170.0),
        (1060.0, 240.0),
        (1100.0, 380.0),
        (1040.0, 520.0),
        (905.0, 615.0),
        (760.0, 650.0),
        (600.0, 660.0),
        (440.0, 645.0),
        (310.0, 600.0),
        (225.0, 500.0),
    ]
    return _build_track_from_control_points(control_points, width=104.0)


def create_grand_endurance_track() -> Track:
    """Large circuit, wider than the window. Mixes long sweepers, a long
    straight, hairpins, and a flowing esses section — both directions.
    """
    control_points = [
        (200.0, 700.0),
        (240.0, 500.0),
        (340.0, 340.0),
        (500.0, 250.0),
        (700.0, 220.0),
        (920.0, 230.0),
        (1140.0, 250.0),
        (1360.0, 230.0),
        (1580.0, 210.0),
        (1780.0, 260.0),
        (1920.0, 390.0),
        (1960.0, 560.0),
        (1880.0, 720.0),
        (1720.0, 810.0),
        (1560.0, 790.0),
        (1460.0, 690.0),
        (1500.0, 560.0),
        (1620.0, 480.0),
        (1540.0, 380.0),
        (1380.0, 410.0),
        (1260.0, 530.0),
        (1280.0, 690.0),
        (1380.0, 830.0),
        (1320.0, 980.0),
        (1140.0, 1040.0),
        (960.0, 1010.0),
        (860.0, 880.0),
        (920.0, 740.0),
        (800.0, 680.0),
        (660.0, 740.0),
        (640.0, 890.0),
        (540.0, 1020.0),
        (380.0, 1080.0),
        (220.0, 1020.0),
        (140.0, 880.0),
    ]
    return _build_track_from_control_points(control_points, width=150.0)


# Registry is ordered by difficulty: basic -> intermediate -> advanced. The
# curriculum in airacer.rl selects stages by slicing this ordering, so keep
# any new entries in the right difficulty group.
TRACK_REGISTRY: Dict[str, Callable[[], Track]] = {
    # basic
    "basic_oval": create_basic_oval_track,
    "basic_rounded": create_basic_rounded_track,
    # intermediate
    "intermediate_snake": create_intermediate_snake_track,
    "intermediate_hairpin": create_intermediate_hairpin_track,
    # advanced (the original three)
    "basic_loop": create_test_track,
    "technical_sprint": create_technical_sprint_track,
    "grand_endurance": create_grand_endurance_track,
}

BASIC_TRACK_NAMES: Tuple[str, ...] = ("basic_oval", "basic_rounded")
INTERMEDIATE_TRACK_NAMES: Tuple[str, ...] = (
    "intermediate_snake",
    "intermediate_hairpin",
)
ADVANCED_TRACK_NAMES: Tuple[str, ...] = (
    "basic_loop",
    "technical_sprint",
    "grand_endurance",
)

REVERSE_SUFFIX = "_rev"
DEFAULT_TRACK_NAME = "basic_oval"


def reverse_track(track: Track) -> Track:
    """Produce a direction-flipped variant of ``track``: the same physical
    geometry (walls unchanged) traversed in the opposite direction. From the
    car's perspective it still feels like driving forwards on a familiar
    loop — only the sequence of left/right turns is mirrored.
    """
    reversed_centerline = tuple(reversed(track.centerline_points))
    spawn_position = reversed_centerline[0]
    if len(reversed_centerline) >= 2:
        spawn_heading = math.atan2(
            reversed_centerline[1][1] - reversed_centerline[0][1],
            reversed_centerline[1][0] - reversed_centerline[0][0],
        )
    else:
        spawn_heading = track.spawn_heading + math.pi
    centerline_segments = tuple(_polyline_segments(reversed_centerline))
    centerline_segment_lengths = tuple(
        segment_length_between(start, end) for start, end in centerline_segments
    )
    return Track(
        outer_points=track.outer_points,
        inner_points=track.inner_points,
        centerline_points=reversed_centerline,
        spawn_position=spawn_position,
        spawn_heading=spawn_heading,
        wall_segments=track.wall_segments,
        centerline_segments=centerline_segments,
        centerline_segment_lengths=centerline_segment_lengths,
        centerline_total_length=track.centerline_total_length,
        wall_index=track.wall_index,
    )


def _base_track_name(name: str) -> Tuple[str, bool]:
    if name.endswith(REVERSE_SUFFIX):
        return name[: -len(REVERSE_SUFFIX)], True
    return name, False


def available_track_names(*, include_reverse: bool = True) -> List[str]:
    base = list(TRACK_REGISTRY.keys())
    if not include_reverse:
        return base
    return base + [name + REVERSE_SUFFIX for name in base]


def basic_track_names(*, include_reverse: bool = True) -> List[str]:
    names = list(BASIC_TRACK_NAMES)
    if include_reverse:
        names += [name + REVERSE_SUFFIX for name in BASIC_TRACK_NAMES]
    return names


def intermediate_track_names(*, include_reverse: bool = True) -> List[str]:
    names = list(INTERMEDIATE_TRACK_NAMES)
    if include_reverse:
        names += [name + REVERSE_SUFFIX for name in INTERMEDIATE_TRACK_NAMES]
    return names


def advanced_track_names(*, include_reverse: bool = True) -> List[str]:
    names = list(ADVANCED_TRACK_NAMES)
    if include_reverse:
        names += [name + REVERSE_SUFFIX for name in ADVANCED_TRACK_NAMES]
    return names


def create_track(name: str) -> Track:
    base, is_reverse = _base_track_name(name)
    if base not in TRACK_REGISTRY:
        raise KeyError(
            "unknown track %r, available: %s"
            % (name, ", ".join(available_track_names()))
        )
    track = TRACK_REGISTRY[base]()
    if is_reverse:
        track = reverse_track(track)
    return track


def point_in_track(point: Point, track: Track) -> bool:
    return _point_in_polygon(point, track.outer_points) and not _point_in_polygon(
        point, track.inner_points
    )


def distance_point_to_segment(point: Point, segment: Segment) -> float:
    px, py = point
    (ax, ay), (bx, by) = segment
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_length_sq = abx * abx + aby * aby
    if ab_length_sq == 0.0:
        return math.hypot(px - ax, py - ay)

    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_length_sq))
    closest_x = ax + abx * t
    closest_y = ay + aby * t
    return math.hypot(px - closest_x, py - closest_y)


def nearest_wall_distance(point: Point, track: Track) -> float:
    best_distance, _, _ = nearest_wall_contact(point, track)
    return best_distance


def nearest_wall_contact(point: Point, track: Track):
    best_distance = float("inf")
    best_segment = track.wall_segments[0]
    best_projected = track.wall_segments[0][0]
    for segment in _nearest_wall_candidates(point, track):
        distance, _, projected = project_point_to_segment(point, segment)
        if distance < best_distance:
            best_distance = distance
            best_segment = segment
            best_projected = projected
    return best_distance, best_segment, best_projected


def raycast_walls(origin: Point, angle: float, max_length: float, track: Track):
    direction = (math.cos(angle), math.sin(angle))
    closest_distance = max_length
    hit_point = (
        origin[0] + direction[0] * max_length,
        origin[1] + direction[1] * max_length,
    )
    found_hit = False

    for segment in _raycast_wall_candidates(origin, direction, max_length, track):
        intersection = ray_segment_intersection(origin, direction, segment)
        if intersection is None:
            continue
        distance, point = intersection
        if 0.0 <= distance <= closest_distance:
            closest_distance = distance
            hit_point = point
            found_hit = True

    return found_hit, closest_distance, hit_point


def _build_wall_spatial_index(segments: Sequence[Segment]) -> WallSpatialIndex:
    cell_size = WALL_GRID_CELL_SIZE
    cells: Dict[GridCell, List[int]] = {}
    bounds: List[Tuple[float, float, float, float]] = []
    min_cell_x = min_cell_y = 10**9
    max_cell_x = max_cell_y = -10**9

    for index, segment in enumerate(segments):
        (ax, ay), (bx, by) = segment
        min_x = min(ax, bx)
        max_x = max(ax, bx)
        min_y = min(ay, by)
        max_y = max(ay, by)
        bounds.append((min_x, min_y, max_x, max_y))

        x0 = _grid_coord(min_x, cell_size)
        x1 = _grid_coord(max_x, cell_size)
        y0 = _grid_coord(min_y, cell_size)
        y1 = _grid_coord(max_y, cell_size)
        min_cell_x = min(min_cell_x, x0)
        max_cell_x = max(max_cell_x, x1)
        min_cell_y = min(min_cell_y, y0)
        max_cell_y = max(max_cell_y, y1)
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                cells.setdefault((cx, cy), []).append(index)

    frozen_cells = {cell: tuple(indices) for cell, indices in cells.items()}
    if not segments:
        min_cell_x = max_cell_x = min_cell_y = max_cell_y = 0
    return WallSpatialIndex(
        cell_size=cell_size,
        cells=frozen_cells,
        segment_bounds=tuple(bounds),
        min_cell_x=min_cell_x,
        max_cell_x=max_cell_x,
        min_cell_y=min_cell_y,
        max_cell_y=max_cell_y,
    )


def _nearest_wall_candidates(point: Point, track: Track) -> Iterable[Segment]:
    index = track.wall_index
    px, py = point
    max_radius = max(
        (index.max_cell_x - index.min_cell_x + 2) * index.cell_size,
        (index.max_cell_y - index.min_cell_y + 2) * index.cell_size,
    )
    radius = index.cell_size
    best_distance = float("inf")
    best_indices: set[int] = set()

    while radius <= max_radius:
        candidates = _segments_in_box(
            px - radius,
            py - radius,
            px + radius,
            py + radius,
            index,
        )
        if candidates:
            best_indices = candidates
            for segment_index in candidates:
                distance = distance_point_to_segment(
                    point, track.wall_segments[segment_index]
                )
                if distance < best_distance:
                    best_distance = distance
            if best_distance <= radius:
                break
        radius *= 2.0

    if not best_indices:
        return track.wall_segments
    return (track.wall_segments[i] for i in best_indices)


def _raycast_wall_candidates(
    origin: Point,
    direction: Point,
    max_length: float,
    track: Track,
) -> Iterable[Segment]:
    index = track.wall_index
    candidate_indices: set[int] = set()
    for cell in _ray_cells(origin, direction, max_length, index.cell_size):
        candidate_indices.update(index.cells.get(cell, ()))

    if not candidate_indices:
        return track.wall_segments
    return (track.wall_segments[i] for i in candidate_indices)


def _ray_cells(
    origin: Point,
    direction: Point,
    max_length: float,
    cell_size: float,
) -> Iterable[GridCell]:
    ox, oy = origin
    dx, dy = direction
    cx = _grid_coord(ox, cell_size)
    cy = _grid_coord(oy, cell_size)
    end_x = ox + dx * max_length
    end_y = oy + dy * max_length
    end_cx = _grid_coord(end_x, cell_size)
    end_cy = _grid_coord(end_y, cell_size)

    step_x = 1 if dx > 0.0 else -1 if dx < 0.0 else 0
    step_y = 1 if dy > 0.0 else -1 if dy < 0.0 else 0
    t_delta_x = cell_size / abs(dx) if dx else float("inf")
    t_delta_y = cell_size / abs(dy) if dy else float("inf")

    if step_x > 0:
        next_x = (cx + 1) * cell_size
        t_max_x = (next_x - ox) / dx
    elif step_x < 0:
        next_x = cx * cell_size
        t_max_x = (next_x - ox) / dx
    else:
        t_max_x = float("inf")

    if step_y > 0:
        next_y = (cy + 1) * cell_size
        t_max_y = (next_y - oy) / dy
    elif step_y < 0:
        next_y = cy * cell_size
        t_max_y = (next_y - oy) / dy
    else:
        t_max_y = float("inf")

    t = 0.0
    limit = 1 + abs(end_cx - cx) + abs(end_cy - cy) + 4
    for _ in range(limit):
        yield cx, cy
        if cx == end_cx and cy == end_cy:
            break
        if t_max_x < t_max_y:
            cx += step_x
            t = t_max_x
            t_max_x += t_delta_x
        else:
            cy += step_y
            t = t_max_y
            t_max_y += t_delta_y
        if t > max_length:
            break


def _segments_in_box(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    index: WallSpatialIndex,
) -> set[int]:
    candidates: set[int] = set()
    x0 = _grid_coord(min_x, index.cell_size)
    x1 = _grid_coord(max_x, index.cell_size)
    y0 = _grid_coord(min_y, index.cell_size)
    y1 = _grid_coord(max_y, index.cell_size)
    for cx in range(x0, x1 + 1):
        for cy in range(y0, y1 + 1):
            candidates.update(index.cells.get((cx, cy), ()))
    return candidates


def _grid_coord(value: float, cell_size: float) -> int:
    return math.floor(value / cell_size)


def sample_centerline_ahead(track: Track, start_progress: float, arc_distance: float):
    """Return (point, tangent_angle) at ``arc_distance`` ahead of ``start_progress`` along the closed centerline."""

    total = track.centerline_total_length
    segments = track.centerline_segments
    segment_lengths = track.centerline_segment_lengths

    if total == 0.0 or not segments:
        start, end = segments[0] if segments else ((0.0, 0.0), (1.0, 0.0))
        return start, math.atan2(end[1] - start[1], end[0] - start[0])

    target_arc = (start_progress * total + arc_distance) % total
    traversed = 0.0
    for index, segment in enumerate(segments):
        length = segment_lengths[index]
        if traversed + length >= target_arc:
            start, end = segment
            t = 0.0 if length == 0.0 else (target_arc - traversed) / length
            point = (
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
            )
            tangent = math.atan2(end[1] - start[1], end[0] - start[0])
            return point, tangent
        traversed += length

    start, end = segments[-1]
    return end, math.atan2(end[1] - start[1], end[0] - start[0])


def progress_on_centerline(point: Point, track: Track):
    segments = track.centerline_segments
    segment_lengths = track.centerline_segment_lengths
    total_length = track.centerline_total_length
    best_distance = float("inf")
    best_progress = 0.0
    traversed = 0.0

    for index, segment in enumerate(segments):
        projection = project_point_to_segment(point, segment)
        distance = projection[0]
        t = projection[1]
        segment_length = segment_lengths[index]
        if distance < best_distance:
            best_distance = distance
            best_progress = traversed + segment_length * t
        traversed += segment_length

    normalized = 0.0 if total_length == 0.0 else best_progress / total_length
    return normalized, best_distance


def polyline_length(points: Sequence[Point]) -> float:
    total = 0.0
    for start, end in _polyline_segments(points):
        total += segment_length_between(start, end)
    return total


def project_point_to_segment(point: Point, segment: Segment):
    px, py = point
    (ax, ay), (bx, by) = segment
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_length_sq = abx * abx + aby * aby
    if ab_length_sq == 0.0:
        return math.hypot(px - ax, py - ay), 0.0, (ax, ay)

    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_length_sq))
    projected = (ax + abx * t, ay + aby * t)
    distance = math.hypot(px - projected[0], py - projected[1])
    return distance, t, projected


def segment_length_between(start: Point, end: Point) -> float:
    return math.hypot(end[0] - start[0], end[1] - start[1])


def _polygon_segments(points: Sequence[Point]) -> List[Segment]:
    segments: List[Segment] = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        segments.append((start, end))
    return segments


def _polyline_segments(points: Sequence[Point]) -> List[Segment]:
    segments: List[Segment] = []
    for index in range(len(points) - 1):
        segments.append((points[index], points[index + 1]))
    return segments


def _chaikin_closed(points: Sequence[Point], iterations: int):
    current = list(points)
    for _ in range(iterations):
        refined: List[Point] = []
        for index, start in enumerate(current):
            end = current[(index + 1) % len(current)]
            refined.append(
                (
                    start[0] * 0.75 + end[0] * 0.25,
                    start[1] * 0.75 + end[1] * 0.25,
                )
            )
            refined.append(
                (
                    start[0] * 0.25 + end[0] * 0.75,
                    start[1] * 0.25 + end[1] * 0.75,
                )
            )
        current = refined
    return current


def _offset_closed_loop(points: Sequence[Point], offset: float):
    source = list(points)
    result: List[Point] = []
    count = len(source)
    for index in range(count):
        prev_point = source[index - 1]
        point = source[index]
        next_point = source[(index + 1) % count]
        normal_x, normal_y = _vertex_normal(prev_point, point, next_point)
        result.append((point[0] + normal_x * offset, point[1] + normal_y * offset))
    return result


def _segment_intersection_point(a1: Point, a2: Point, b1: Point, b2: Point):
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    eps = 1e-9
    if not (eps < t < 1.0 - eps and eps < u < 1.0 - eps):
        return None
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _signed_polygon_area(points: Sequence[Point]) -> float:
    total = 0.0
    count = len(points)
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        total += x1 * y2 - x2 * y1
    return 0.5 * total


def _remove_self_intersections(points: Sequence[Point]) -> List[Point]:
    loop: List[Point] = list(points)
    original_area = abs(_signed_polygon_area(loop))
    if original_area == 0.0:
        return loop
    knot_area_ratio = 0.02
    for _ in range(256):
        count = len(loop)
        if count < 4:
            return loop
        hit_info = None
        for i in range(count):
            a1 = loop[i]
            a2 = loop[(i + 1) % count]
            for j in range(i + 2, count):
                if i == 0 and j == count - 1:
                    continue
                b1 = loop[j]
                b2 = loop[(j + 1) % count]
                hit = _segment_intersection_point(a1, a2, b1, b2)
                if hit is not None:
                    hit_info = (i, j, hit)
                    break
            if hit_info is not None:
                break
        if hit_info is None:
            return loop
        i, j, hit = hit_info
        loop_outer = loop[: i + 1] + [hit] + loop[j + 1 :]
        loop_inner = [hit] + loop[i + 1 : j + 1]
        area_outer = abs(_signed_polygon_area(loop_outer))
        area_inner = abs(_signed_polygon_area(loop_inner))
        smaller_area = min(area_outer, area_inner)
        if smaller_area / original_area > knot_area_ratio:
            return list(points)
        if area_outer >= area_inner:
            loop = loop_outer
        else:
            loop = loop_inner
    return loop


def _vertex_normal(prev_point: Point, point: Point, next_point: Point):
    prev_dx = point[0] - prev_point[0]
    prev_dy = point[1] - prev_point[1]
    next_dx = next_point[0] - point[0]
    next_dy = next_point[1] - point[1]

    prev_len = math.hypot(prev_dx, prev_dy) or 1.0
    next_len = math.hypot(next_dx, next_dy) or 1.0

    prev_nx = -prev_dy / prev_len
    prev_ny = prev_dx / prev_len
    next_nx = -next_dy / next_len
    next_ny = next_dx / next_len

    normal_x = prev_nx + next_nx
    normal_y = prev_ny + next_ny
    normal_len = math.hypot(normal_x, normal_y)
    if normal_len < 1e-9:
        return next_nx, next_ny
    return normal_x / normal_len, normal_y / normal_len


def _closed_polyline(points: Sequence[Point]):
    if not points:
        return []
    return list(points) + [points[0]]


def _rotate_loop_start_near_point(points: Sequence[Point], target: Point):
    if not points:
        return []
    best_index = min(
        range(len(points)),
        key=lambda index: math.hypot(
            points[index][0] - target[0], points[index][1] - target[1]
        ),
    )
    return list(points[best_index:]) + list(points[:best_index])


def ray_segment_intersection(origin: Point, direction: Point, segment: Segment):
    ox, oy = origin
    dx, dy = direction
    (ax, ay), (bx, by) = segment
    sx = bx - ax
    sy = by - ay

    denominator = dx * sy - dy * sx
    if abs(denominator) < 1e-9:
        return None

    qpx = ax - ox
    qpy = ay - oy
    ray_distance = (qpx * sy - qpy * sx) / denominator
    segment_t = (qpx * dy - qpy * dx) / denominator

    if ray_distance < 0.0 or not (0.0 <= segment_t <= 1.0):
        return None

    point = (ox + dx * ray_distance, oy + dy * ray_distance)
    return ray_distance, point


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside
