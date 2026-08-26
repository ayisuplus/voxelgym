"""Metric adapter for task and oracle-expert world access.

Task geometry is authored in canonical one-meter voxels.  ``PyWorld`` keeps
its high-throughput compatibility methods in engine-cell units, so this
module is the single conversion boundary used after world creation.  Spatial
scale is immutable, integral, and means ``cells_per_meter``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


MetricCell = tuple[int, int, int]
EngineCell = tuple[int, int, int]
MetricPoint = tuple[float, float, float]


def _read_world_scale(world) -> int:
    scale_accessor = getattr(world, "scale", None)
    if callable(scale_accessor):
        raw = scale_accessor()
    elif scale_accessor is not None:
        raw = scale_accessor
    else:
        oracle_state = getattr(world, "oracle_state", None)
        raw = oracle_state().get("scale", 1.0) if callable(oracle_state) else 1.0
    scale = float(raw)
    rounded = int(round(scale))
    if not math.isfinite(scale) or rounded < 1 or not math.isclose(scale, rounded):
        raise ValueError(f"task metric adapter requires integral scale >= 1, got {raw!r}")
    return rounded


def world_scale(world) -> int:
    """Return immutable cells-per-meter, with a scale-1 legacy fallback."""

    return _read_world_scale(world)


def metric_cell_volume(world, at: Sequence[int]) -> tuple[EngineCell, ...]:
    """All engine cells occupied by one canonical meter voxel.

    Integer multiplication and half-open ranges preserve the expected floor
    mapping on the negative side of the origin (for example, meter x=-2 at
    scale 2 maps to engine cells -4 and -3).
    """

    if len(at) != 3:
        raise ValueError("metric cell coordinate must have exactly three components")
    x, y, z = (int(value) for value in at)
    scale = world_scale(world)
    return tuple(
        (cx, cy, cz)
        for cx in range(x * scale, (x + 1) * scale)
        for cy in range(y * scale, (y + 1) * scale)
        for cz in range(z * scale, (z + 1) * scale)
    )


def metric_set_block(world, at: Sequence[int], cell: int) -> None:
    """Fill an entire canonical meter voxel with one raw block value."""

    for x, y, z in metric_cell_volume(world, at):
        world.set_block(x, y, z, int(cell))


def metric_block_values(world, at: Sequence[int]) -> tuple[int, ...]:
    """Raw values for the whole logical voxel in deterministic XYZ order."""

    return tuple(world.get_block(x, y, z) for x, y, z in metric_cell_volume(world, at))


def metric_get_block(world, at: Sequence[int]) -> int:
    """Return the first occupied subcell, or air when the whole voxel is air.

    Refined worlds can temporarily contain a partially changed logical voxel.
    Choosing the first non-air value makes interaction experts finish clearing
    the full meter volume instead of treating the first removed subcell as the
    whole block.  Tests that need quantification use the explicit any/all
    predicates below.
    """

    values = metric_block_values(world, at)
    return next((value for value in values if value & 0xFFF), values[0])


def metric_any_block_is(world, at: Sequence[int], block_id: int) -> bool:
    """Whether any refined subcell has the requested registry ID."""

    expected = int(block_id) & 0xFFF
    return any(value & 0xFFF == expected for value in metric_block_values(world, at))


def metric_all_blocks_are(world, at: Sequence[int], block_id: int) -> bool:
    """Whether every refined subcell has the requested registry ID."""

    expected = int(block_id) & 0xFFF
    return all(value & 0xFFF == expected for value in metric_block_values(world, at))


def metric_set_cell_interventions(
    world, at: Sequence[int], cell: int
) -> list[dict[str, object]]:
    """Expand a typed logical set-cell intervention to exact engine cells."""

    return [
        {"kind": "set_cell", "at": [x, y, z], "cell": int(cell)}
        for x, y, z in metric_cell_volume(world, at)
    ]


def metric_set_cells_interventions(
    world, cells: Iterable[Sequence[int]], cell: int
) -> list[dict[str, object]]:
    """Expand multiple logical meter voxels in stable caller order."""

    return [
        spec
        for at in cells
        for spec in metric_set_cell_interventions(world, at, cell)
    ]


def agent_position_meters(world) -> MetricPoint:
    """Read the compatibility cell-space agent position as meters."""

    scale = world_scale(world)
    x, y, z = world.agent_pos()
    return (float(x) / scale, float(y) / scale, float(z) / scale)


def teleport_meters(world, position: Sequence[float]) -> None:
    """Teleport to a metric world position through the cell-space API."""

    if len(position) != 3:
        raise ValueError("metric position must have exactly three components")
    scale = world_scale(world)
    world.teleport(*(float(value) * scale for value in position))


def metric_surface_y(world, x: int, z: int) -> int:
    """Highest occupied canonical meter-cell Y across a meter column."""

    scale = world_scale(world)
    raw = max(
        world.surface_y(cx, cz)
        for cx in range(int(x) * scale, (int(x) + 1) * scale)
        for cz in range(int(z) * scale, (int(z) + 1) * scale)
    )
    return -1 if raw < 0 else raw // scale


def engine_cell_to_metric_cell(world, at: Sequence[int]) -> MetricCell:
    """Map an engine cell to its containing canonical meter voxel."""

    scale = world_scale(world)
    x, y, z = (int(value) for value in at)
    return (x // scale, y // scale, z // scale)


def metric_find_blocks(world, block_id: int, radius_meters: int) -> list[MetricCell]:
    """Find logical meter voxels while querying the native cell-space API."""

    scale = world_scale(world)
    raw = world.find_blocks(int(block_id), int(math.ceil(radius_meters * scale)))
    found: list[MetricCell] = []
    seen: set[MetricCell] = set()
    for at in raw:
        logical = engine_cell_to_metric_cell(world, at)
        if logical not in seen:
            seen.add(logical)
            found.append(logical)
    return found


def metric_crosshair(world):
    """Convert the native crosshair cell coordinate to the metric grid."""

    hit = world.crosshair()
    if hit is None:
        return None
    return (engine_cell_to_metric_cell(world, hit[0]), hit[1])


def metric_drops_of(world, item: int) -> list[MetricPoint]:
    """Convert dynamic drop positions from continuous cells to meters."""

    scale = world_scale(world)
    return [tuple(float(value) / scale for value in drop) for drop in world.drops_of(item)]


def metric_furnace_state(world, at: Sequence[int]):
    """Read the representative native furnace inside a logical voxel."""

    return world.furnace_state(*metric_cell_volume(world, at)[0])
