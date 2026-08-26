"""Canonical serialized intervention vocabulary shared by env and bundles."""

from __future__ import annotations

from collections.abc import Mapping
import math
import operator
from typing import Any


def _integer(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be an integer in [{minimum}, {maximum}]"
        ) from exc
    if not minimum <= normalized <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(normalized)


def _triplet(value: Any, *, name: str, integer: bool) -> list[int] | list[float]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must contain exactly three coordinates")
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly three coordinates") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    if integer:
        return [
            _integer(
                component,
                name=f"{name}[{index}]",
                minimum=-(1 << 31),
                maximum=(1 << 31) - 1,
            )
            for index, component in enumerate(values)
        ]
    normalized = []
    for index, component in enumerate(values):
        if isinstance(component, bool):
            raise ValueError(f"{name}[{index}] must be a finite number")
        try:
            number = float(component)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be a finite number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be a finite number")
        normalized.append(number)
    return normalized


def canonical_intervention_spec(spec: Any) -> dict[str, Any]:
    """Validate one intervention and return its stable JSON-safe shape.

    The native binding accepts ``type`` as a historical alias for ``kind``.
    Bundles always persist ``kind`` and only fields owned by the selected
    variant, so replay does not depend on Python object types or extra keys.
    """

    if not isinstance(spec, Mapping):
        raise ValueError("intervention spec must be a mapping")
    kind = spec.get("kind")
    alias = spec.get("type")
    if kind is None:
        kind = alias
    elif alias is not None and alias != kind:
        raise ValueError("intervention 'kind' and 'type' tags disagree")
    if not isinstance(kind, str):
        raise ValueError("intervention requires a string 'kind' tag")

    if kind == "set_cell":
        return {
            "kind": kind,
            "at": _triplet(spec.get("at"), name="set_cell 'at'", integer=True),
            "cell": _integer(
                spec.get("cell"), name="set_cell 'cell'", minimum=0, maximum=65535
            ),
        }
    if kind == "teleport_agent":
        position = _triplet(
            spec.get("position"), name="teleport_agent 'position'", integer=False
        )
        # The engine converts a world position to an i32 cell coordinate.
        if any(
            not -(1 << 31) <= math.floor(value) <= (1 << 31) - 1
            for value in position
        ):
            raise ValueError("teleport_agent position must fit world coordinates")
        return {"kind": kind, "position": position}
    if kind == "set_agent_velocity":
        return {
            "kind": kind,
            "velocity": _triplet(
                spec.get("velocity"),
                name="set_agent_velocity 'velocity'",
                integer=False,
            ),
        }
    if kind == "give_item":
        return {
            "kind": kind,
            "item": _integer(
                spec.get("item"), name="give_item 'item'", minimum=0, maximum=65535
            ),
            "count": _integer(
                spec.get("count"),
                name="give_item 'count'",
                minimum=0,
                maximum=65535,
            ),
        }
    if kind == "swap_to_hotbar":
        return {
            "kind": kind,
            "item": _integer(
                spec.get("item"),
                name="swap_to_hotbar 'item'",
                minimum=0,
                maximum=65535,
            ),
        }
    raise ValueError(f"unknown intervention kind {kind!r}")


def canonical_interventions(specs: Any) -> tuple[dict[str, Any], ...]:
    """Return an immutable sequence of detached canonical specs."""

    if isinstance(specs, (str, bytes, bytearray, Mapping)):
        raise ValueError("interventions must be a sequence of specs")
    try:
        return tuple(canonical_intervention_spec(spec) for spec in specs)
    except TypeError as exc:
        raise ValueError("interventions must be a sequence of specs") from exc
