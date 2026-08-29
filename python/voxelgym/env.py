"""gymnasium environment over the Rust voxel core.

Observation/action spaces follow the plan contract literally.

Cell encoding in `voxels`: raw u16 cell = (state << 12) | block_id.
  - low 12 bits: block id (registry truth table)
  - high 4 bits: state (fluid level 0..15, wire power 0..15, door/lever bit)

`raycast` distance is in centimetres (450 = 4.5-metre reach cap, no target).
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
import math
import operator
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import voxelgym_rs as rs

from .interventions import canonical_interventions
from .task_state import EnvSnapshot, RewardOutcome, StatefulTask, clone_sensor_cache

ACTION_KEYS = ("move", "jump", "sneak", "yaw", "pitch", "mine", "place", "use", "hotbar", "craft")

RENDER_RES = 128


def _canonical_physics_config(physics: dict[str, float] | None) -> dict[str, float] | None:
    if physics is None:
        return None
    if not isinstance(physics, dict):
        raise TypeError("physics must be a mapping or None")
    result: dict[str, float] = {}
    for key, raw_value in physics.items():
        if not isinstance(key, str):
            raise TypeError("physics field names must be strings")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(f"physics field {key!r} must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"physics field {key!r} must be finite")
        result[key] = value
    return dict(sorted(result.items()))


def action_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "move": spaces.Discrete(5),      # 0 idle, 1 fwd, 2 back, 3 left, 4 right
            "jump": spaces.Discrete(2),
            "sneak": spaces.Discrete(2),
            "yaw": spaces.Discrete(24),      # absolute heading, 15 deg buckets
            "pitch": spaces.Discrete(9),     # absolute pitch -60..+60, 15 deg buckets
            "mine": spaces.Discrete(2),
            "place": spaces.Discrete(2),
            "use": spaces.Discrete(2),
            "hotbar": spaces.Discrete(9),
            "craft": spaces.Discrete(8),     # recipe id, 0 = noop
        }
    )


def random_action(rng: np.random.Generator, zero: tuple[str, ...] = ()) -> dict[str, int]:
    """Uniform-random action dict drawn from action_space() — the single
    source for demos and epsilon-mixing, so the bounds can never drift from
    the contract. Fields named in `zero` are forced to 0 WITHOUT consuming
    rng draws (keeps per-episode streams comparable across configurations).
    """
    return {
        k: 0 if k in zero else int(rng.integers(0, space.n))
        for k, space in action_space().items()
    }


def observation_space(
    render: bool = False,
    lidar: dict | None = None,
    spacetime: bool = False,
) -> spaces.Dict:
    d: dict[str, spaces.Space] = {
        "voxels": spaces.Box(0, 65535, (21, 11, 21), dtype=np.uint16),
        "inventory": spaces.Box(0, 65535, (36, 2), dtype=np.uint16),
        "pose": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
        "raycast": spaces.Box(0, 65535, (2,), dtype=np.uint16),
    }
    if render:
        d.update(
            {
                "rgb": spaces.Box(0, 255, (RENDER_RES, RENDER_RES, 3), dtype=np.uint8),
                "depth": spaces.Box(0, np.inf, (RENDER_RES, RENDER_RES), dtype=np.float16),
                "seg": spaces.Box(0, 65535, (RENDER_RES, RENDER_RES), dtype=np.uint16),
                # per-pixel surface normal (unit axis, f32); [0,0,0] on sky.
                # pixel vector = [r, g, b, depth, block_id, nx, ny, nz]
                "normals": spaces.Box(-1, 1, (RENDER_RES, RENDER_RES, 3), dtype=np.float32),
            }
        )
    if lidar:
        c, a = lidar["channels"], lidar["azimuth"]
        d.update(
            {
                # range image: row = elevation channel, col = azimuth step
                # (the standard RangeNet++/PointPillars input layout)
                "lidar_range": spaces.Box(0, np.inf, (c, a), dtype=np.float32),
                "lidar_intensity": spaces.Box(0, 1, (c, a), dtype=np.float32),
                "lidar_seg": spaces.Box(0, 65535, (c, a), dtype=np.uint16),
            }
        )
    if spacetime:
        d.update(
            {
                # tick, elapsed seconds, remaining ticks/seconds, and the
                # render/LiDAR sample times (negative means unavailable).
                "clock": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float64),
                "velocity": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                # metric translation, yaw delta, pitch delta, elapsed time.
                "egomotion": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
                # frame id, cells per meter, meters per cell.
                "spatial_meta": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                "sensor_age": spaces.Box(-1, np.inf, (2,), dtype=np.float32),
                # occupied cells immediately west/east/below/above/north/south
                # in the already-visible local voxel window.
                "local_relations": spaces.MultiBinary(6),
            }
        )
    return spaces.Dict(d)


class VoxelGymEnv(gym.Env):
    """Single voxel world. `task` is a duck-typed Task (M2/M3): it provides
    preset/horizon/scenario(rng)/semantic_regions(rng)/on_reset(world)/
    reward_outcome(world, events=()), plus
    optional interventions_before_step/state_dict hooks; this env handles sim
    + obs plumbing.

    render: False | True | int N — render every N ticks (True == 1); frames
    between renders reuse the previous frame (M4).
    lidar: None | dict — spinning multi-beam LiDAR channel. Keys:
    channels, azimuth, min_elev, max_elev, max_range, and optional
    every (scan every N ticks, default 1), noise_sigma, dropout_p.
    scale: cells per meter (1.0 default; 2.0 = 0.5 m cells — same physical
    world, finer voxels; world height becomes 128*scale).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: Any = None,
        preset: str | None = None,
        seed: int = 0,
        render: Literal[False, True] | int = False,
        lidar: dict | None = None,
        scale: float = 1.0,
        dt_numerator: int = 1,
        dt_denominator: int = 20,
        spacetime: bool = False,
        semantic_regions: list[tuple[int, ...]] | None = None,
        physics: dict[str, float] | None = None,
    ):
        super().__init__()
        _validate_task_reward_contract(task)
        self._task = task
        self._preset = preset or getattr(task, "preset", None) or "default"
        self._seed0 = seed
        self._scale = float(scale)
        self._dt_numerator = int(dt_numerator)
        self._dt_denominator = int(dt_denominator)
        self._spacetime = bool(spacetime)
        self._semantic_regions = deepcopy(semantic_regions)
        self._physics = _canonical_physics_config(physics)
        if render is True:
            render = 1
        self._render_every = int(render) if render else 0
        self._lidar = dict(lidar) if lidar else None
        self._lidar_every = int(self._lidar.get("every", 1)) if self._lidar else 0
        self._last_scan: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._last_scan_tick: int | None = None
        self.action_space = action_space()
        self.observation_space = observation_space(
            self._render_every > 0, self._lidar, self._spacetime
        )
        self._w: rs.PyWorld | None = None
        self._episode_seed = seed
        self._last_frames: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._last_frame_tick: int | None = None
        self._terminated = False
        self._truncated = False
        self._last_reward = RewardOutcome()
        self._last_trace: dict[str, Any] | None = None
        self._previous_pose: tuple[float, ...] | None = None
        self._intervention_cursor = 0

    @property
    def world(self) -> rs.PyWorld:
        assert self._w is not None, "call reset() first"
        return self._w

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._episode_seed = self._seed0 if seed is None else seed
        scenario = None
        semantic_regions = deepcopy(self._semantic_regions)
        if self._task is not None:
            semantic_fn = getattr(self._task, "semantic_regions", None)
            task_regions = (
                None if semantic_fn is None else semantic_fn(self.np_random)
            )
            if semantic_regions is not None and task_regions is not None:
                raise ValueError(
                    "semantic regions were provided by both the environment and task"
                )
            semantic_regions = (
                semantic_regions if semantic_regions is not None else task_regions
            )
            if semantic_regions is None:
                scenario = self._task.scenario(self.np_random)
        self._w = rs.PyWorld(
            self._episode_seed,
            self._preset,
            scenario,
            scale=self._scale,
            dt_numerator=self._dt_numerator,
            dt_denominator=self._dt_denominator,
            semantic_regions=semantic_regions,
            physics=deepcopy(self._physics),
        )
        self._last_frames = None
        self._last_scan = None
        self._last_frame_tick = None
        self._last_scan_tick = None
        self._terminated = False
        self._truncated = False
        self._last_reward = RewardOutcome()
        self._last_trace = None
        self._previous_pose = None
        self._intervention_cursor = 0
        if self._task is not None:
            self._task.on_reset(self._w, self.np_random)
        observation = self._obs()
        return observation, {
            "seed": self._episode_seed,
            "clock": self.world.clock(),
            "physics_config": deepcopy(self._physics),
            "physics": dict(self.world.physics()),
            "scale": self._scale,
            "sensor_profile": self.sensor_profile(),
        }

    def step(self, action: dict[str, Any]):
        return self._transition(action)

    def step_traced(
        self,
        action: dict[str, Any],
        *,
        trace_level: str = "events",
        branch_id: int = 0,
        interventions: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
    ):
        """Advance a traced transition with optional serialized inputs.

        Explicit interventions occur after the pre-transition boundary and
        before the task's intervention phase and agent action.  Their events
        and deltas are merged into the same oracle-only causal outcome.
        """

        return self._transition(
            action,
            trace_level=trace_level,
            branch_id=branch_id,
            external_interventions=tuple(interventions),
        )

    def _transition(
        self,
        action: dict[str, Any],
        *,
        trace_level: str | None = None,
        branch_id: int = 0,
        external_interventions: tuple[dict[str, Any], ...] = (),
    ):
        w = self.world
        if trace_level not in (None, "off", "events", "full"):
            raise ValueError("trace_level must be one of: off, events, full")
        packed_values: list[int] = []
        for key in ACTION_KEYS:
            raw_value = action[key]
            try:
                value = operator.index(raw_value)
            except TypeError as exc:
                raise TypeError(f"action {key!r} must be an integer") from exc
            action_dimension = self.action_space.spaces[key]
            if not 0 <= value < action_dimension.n:
                raise ValueError(
                    f"action {key!r} must be in [0, {action_dimension.n})"
                )
            packed_values.append(value)
        packed_action = tuple(packed_values)
        try:
            branch_id = operator.index(branch_id)
        except TypeError as exc:
            raise TypeError("branch_id must be an integer") from exc
        if not 0 <= branch_id <= (1 << 64) - 1:
            raise ValueError("branch_id must fit an unsigned 64-bit integer")
        transition_before_hash = w.hash() if trace_level == "full" else None
        intervention_traces: list[dict[str, Any]] = []
        external_intervention_count = len(external_interventions)
        specs: list[dict[str, Any]] = list(external_interventions)
        task_state_before_interventions = None
        if self._task is not None:
            specs_fn = getattr(self._task, "interventions_before_step", None)
            if specs_fn is not None:
                task_state_before_interventions = deepcopy(self._task.state_dict())
                try:
                    task_specs = _call_with_read_only_world(
                        w,
                        lambda read_world: specs_fn(read_world, action),
                        "interventions_before_step",
                    )
                    specs.extend(task_specs or ())
                except Exception:
                    self._task.load_state_dict(task_state_before_interventions)
                    raise

        try:
            # Detach mutable caller/task objects and normalize aliases/numeric
            # wrappers before either validation or mutation.  This exact list
            # is the transition input persisted by Episode Bundle v2.
            specs = list(canonical_interventions(specs))
        except Exception:
            if task_state_before_interventions is not None:
                self._task.load_state_dict(task_state_before_interventions)
            raise

        if specs:
            # One transition owns the whole intervention batch: validation or
            # application failure must leave the common branch point intact.
            try:
                probe = w.fork()
                for offset, spec in enumerate(specs):
                    probe.apply_intervention(
                        spec,
                        trace_level=trace_level or "off",
                        branch_id=branch_id,
                        intervention_id=self._intervention_cursor + offset,
                    )
            except Exception:
                if task_state_before_interventions is not None:
                    self._task.load_state_dict(task_state_before_interventions)
                raise

            world_before_interventions = bytes(w.snapshot())
            trace_state_before_interventions = _capture_native_trace_state(w)
            native_cursor_before_interventions = _capture_native_intervention_cursor(w)
            env_cursor_before_interventions = self._intervention_cursor
            try:
                for spec in specs:
                    intervention_traces.append(
                        w.apply_intervention(
                            spec,
                            trace_level=trace_level or "off",
                            branch_id=branch_id,
                            intervention_id=self._intervention_cursor,
                        )
                    )
                    self._intervention_cursor += 1
            except Exception:
                w.restore(world_before_interventions)
                _restore_native_trace_state(w, trace_state_before_interventions)
                _restore_native_intervention_cursor(
                    w, native_cursor_before_interventions
                )
                self._intervention_cursor = env_cursor_before_interventions
                if task_state_before_interventions is not None:
                    self._task.load_state_dict(task_state_before_interventions)
                raise
        trace = None
        if trace_level is None:
            w.step(packed_action)
        else:
            trace = w.step_traced(
                packed_action, trace_level=trace_level, branch_id=branch_id
            )
            if intervention_traces:
                trace["events"] = [
                    result["event"]
                    for result in intervention_traces
                    if result.get("event") is not None
                ] + trace["events"]
                trace["deltas"] = [
                    delta
                    for result in intervention_traces
                    for delta in result.get("deltas", ())
                ] + trace["deltas"]
                # A Gym transition starts before its intervention phase.
                # Keep the branch point hash common across factual/treatment
                # records; the intervention event/delta explains the mutation.
                trace["before_hash"] = transition_before_hash
        outcome = RewardOutcome()
        if self._task is not None:
            task_before = deepcopy(self._task.state_dict())
            task_fingerprint = _task_state_fingerprint(task_before)
            try:
                outcome = _call_with_read_only_world(
                    w,
                    lambda read_world: _invoke_reward_outcome(
                        self._task.reward_outcome,
                        read_world,
                        ()
                        if trace is None
                        else tuple(deepcopy(trace.get("events", ()))),
                    ),
                    "reward_outcome",
                )
            except Exception:
                self._task.load_state_dict(task_before)
                raise
            if _task_state_fingerprint(self._task.state_dict()) != task_fingerprint:
                self._task.load_state_dict(task_before)
                raise TypeError(
                    "task reward_outcome(world, events=()) mutated Task State; return "
                    "task_state_updates in RewardOutcome instead"
                )
            if not isinstance(outcome, RewardOutcome):
                raise TypeError(
                    "task reward_outcome(world, events=()) must return RewardOutcome"
                )
        # Evidence is selected by the pure task evaluator.  Attaching every
        # event in a busy transition would falsely label unrelated fluid,
        # fire, circuit, or item activity as reward evidence.
        dead = bool(w.dead())
        terminated = dead or outcome.terminated
        if dead and not outcome.terminated:
            outcome = RewardOutcome(
                total=outcome.total,
                components=dict(outcome.components),
                terminated=True,
                termination_reason="agent_dead",
                evidence_event_ids=outcome.evidence_event_ids,
                evidence_labels=outcome.evidence_labels + ("env:agent_dead",),
                task_state_updates=dict(outcome.task_state_updates),
            )
        truncated = False
        horizon = getattr(self._task, "horizon", None)
        if horizon is not None and w.tick() >= horizon:
            truncated = True
            if outcome.termination_reason is None:
                outcome = RewardOutcome(
                    total=outcome.total,
                    components=dict(outcome.components),
                    terminated=outcome.terminated,
                    termination_reason="horizon",
                    evidence_event_ids=outcome.evidence_event_ids,
                    evidence_labels=outcome.evidence_labels + ("env:horizon",),
                    task_state_updates=dict(outcome.task_state_updates),
                )
        if self._task is not None:
            commit_reward = getattr(self._task, "commit_reward", None)
            if commit_reward is None:
                if outcome.task_state_updates:
                    raise TypeError(
                        "task returned state updates but does not implement commit_reward"
                    )
            else:
                commit_reward(outcome)
        self._terminated = terminated
        self._truncated = truncated
        self._last_reward = outcome
        self._last_trace = None if trace is None else deepcopy(trace)
        obs = self._obs()
        info = {
            "tick": w.tick(),
            "clock": w.clock(),
            "reward_outcome": outcome.as_info(),
            "sensor_sample_ticks": {
                "render": self._last_frame_tick,
                "lidar": self._last_scan_tick,
            },
            "interventions": deepcopy(specs),
            "external_intervention_count": external_intervention_count,
        }
        return obs, float(outcome.total), terminated, truncated, info

    def snapshot(self) -> EnvSnapshot:
        """Capture all state required to continue this exact Gym episode."""
        task_state = None
        if self._task is not None:
            if not isinstance(self._task, StatefulTask):
                raise TypeError("task must implement state_dict/load_state_dict to snapshot an episode")
            task_state = deepcopy(self._task.state_dict())
        last_reward = RewardOutcome(
            total=self._last_reward.total,
            components=dict(self._last_reward.components),
            terminated=self._last_reward.terminated,
            termination_reason=self._last_reward.termination_reason,
            evidence_event_ids=tuple(self._last_reward.evidence_event_ids),
            evidence_labels=tuple(self._last_reward.evidence_labels),
            task_state_updates=dict(self._last_reward.task_state_updates),
        )
        native_trace_state = _capture_native_trace_state(self.world)
        return EnvSnapshot(
            world_snapshot=bytes(self.world.snapshot()),
            task_state=task_state,
            np_random_state=deepcopy(self.np_random.bit_generator.state),
            episode_seed=int(self._episode_seed),
            terminated=self._terminated,
            truncated=self._truncated,
            last_reward=last_reward,
            last_frames=clone_sensor_cache(self._last_frames),
            last_scan=clone_sensor_cache(self._last_scan),
            render_sample_tick=self._last_frame_tick,
            lidar_sample_tick=self._last_scan_tick,
            previous_pose=self._previous_pose,
            render_every=self._render_every,
            lidar_config=deepcopy(self._lidar),
            spacetime=self._spacetime,
            last_trace=deepcopy(self._last_trace),
            native_trace_state=native_trace_state,
            native_intervention_cursor=_capture_native_intervention_cursor(
                self.world
            ),
            intervention_cursor=self._intervention_cursor,
            physics_config=deepcopy(self._physics),
        )

    def restore(self, snapshot: EnvSnapshot) -> None:
        """Restore a checkpoint without re-running task generation or reset hooks."""
        if not isinstance(snapshot, EnvSnapshot):
            raise TypeError("snapshot must be an EnvSnapshot")
        candidate_task = None
        if snapshot.task_state is not None:
            if self._task is None or not isinstance(self._task, StatefulTask):
                raise ValueError("snapshot contains task state but this environment has no stateful task")
            candidate_task = deepcopy(self._task)
            candidate_task.load_state_dict(deepcopy(snapshot.task_state))
        elif self._task is not None:
            raise ValueError("snapshot has no task state but this environment has a task")
        candidate_world = rs.PyWorld(
            snapshot.episode_seed,
            self._preset,
            scale=self._scale,
            dt_numerator=self._dt_numerator,
            dt_denominator=self._dt_denominator,
            physics=deepcopy(snapshot.physics_config),
        )
        candidate_world.restore(snapshot.world_snapshot)
        _restore_native_trace_state(candidate_world, snapshot.native_trace_state)
        _restore_native_intervention_cursor(
            candidate_world, snapshot.native_intervention_cursor
        )
        candidate_rng = np.random.default_rng()
        candidate_rng.bit_generator.state = deepcopy(snapshot.np_random_state)
        restored_state = candidate_world.oracle_state()
        restored_clock = restored_state["clock"]
        candidate_scale = float(restored_state["scale"])
        candidate_dt_numerator = int(restored_clock["dt_numerator"])
        candidate_dt_denominator = int(restored_clock["dt_denominator"])
        (
            candidate_render_every,
            candidate_lidar,
            candidate_lidar_every,
            candidate_spacetime,
            candidate_observation_space,
        ) = _validate_snapshot_observation_profile(
            snapshot, world_tick=int(restored_clock["tick"])
        )
        candidate_frames = clone_sensor_cache(snapshot.last_frames)
        candidate_scan = clone_sensor_cache(snapshot.last_scan)

        # Commit only after every restorable layer has validated. Preserve the
        # original task object's identity because experts may hold a reference.
        if candidate_task is not None:
            original_task_state = deepcopy(self._task.state_dict())
            try:
                self._task.load_state_dict(candidate_task.state_dict())
            except Exception:
                self._task.load_state_dict(original_task_state)
                raise
        self._w = candidate_world
        self.np_random.bit_generator.state = deepcopy(candidate_rng.bit_generator.state)
        self._scale = candidate_scale
        self._dt_numerator = candidate_dt_numerator
        self._dt_denominator = candidate_dt_denominator
        self._physics = deepcopy(snapshot.physics_config)
        self._render_every = candidate_render_every
        self._lidar = candidate_lidar
        self._lidar_every = candidate_lidar_every
        self._spacetime = candidate_spacetime
        self.observation_space = candidate_observation_space
        self._episode_seed = int(snapshot.episode_seed)
        self._terminated = bool(snapshot.terminated)
        self._truncated = bool(snapshot.truncated)
        self._last_reward = RewardOutcome(
            total=snapshot.last_reward.total,
            components=dict(snapshot.last_reward.components),
            terminated=snapshot.last_reward.terminated,
            termination_reason=snapshot.last_reward.termination_reason,
            evidence_event_ids=tuple(snapshot.last_reward.evidence_event_ids),
            evidence_labels=tuple(snapshot.last_reward.evidence_labels),
            task_state_updates=dict(snapshot.last_reward.task_state_updates),
        )
        self._last_trace = deepcopy(snapshot.last_trace)
        self._last_frames = candidate_frames
        self._last_scan = candidate_scan
        self._last_frame_tick = snapshot.render_sample_tick
        self._last_scan_tick = snapshot.lidar_sample_tick
        self._previous_pose = snapshot.previous_pose
        self._intervention_cursor = int(snapshot.intervention_cursor)

    def fork(self) -> "VoxelGymEnv":
        """Create an independent complete-environment branch at this boundary."""

        clone = VoxelGymEnv(
            task=deepcopy(self._task),
            preset=self._preset,
            seed=self._episode_seed,
            render=self._render_every,
            lidar=deepcopy(self._lidar),
            scale=self._scale,
            dt_numerator=self._dt_numerator,
            dt_denominator=self._dt_denominator,
            spacetime=self._spacetime,
            semantic_regions=deepcopy(self._semantic_regions),
            physics=deepcopy(self._physics),
        )
        # Gymnasium initializes np_random lazily; restore then replaces its
        # state without invoking scenario/on_reset side effects.
        _ = clone.np_random
        clone.restore(self.snapshot())
        return clone

    def _obs(self) -> dict[str, np.ndarray]:
        w = self.world
        obs = {
            "voxels": w.obs_voxels(),
            "inventory": w.obs_inventory(),
            "pose": w.obs_pose(),
            "raycast": w.obs_raycast(),
        }
        if self._render_every > 0:
            rgb, depth, seg, normals = self._frames()
            obs.update({"rgb": rgb, "depth": depth, "seg": seg, "normals": normals})
        if self._lidar:
            rng_i, inten, seg = self._scan()
            obs.update({"lidar_range": rng_i, "lidar_intensity": inten, "lidar_seg": seg})
        if self._spacetime:
            obs.update(self._spacetime_obs(obs["voxels"]))
        return obs

    def _spacetime_obs(self, voxels: np.ndarray) -> dict[str, np.ndarray]:
        state = self.world.oracle_state()
        clock = state["clock"]
        horizon = getattr(self._task, "horizon", None)
        remaining_ticks = -1 if horizon is None else max(0, int(horizon) - int(clock["tick"]))
        remaining_seconds = (
            -1.0
            if horizon is None
            else remaining_ticks * float(clock["seconds_per_tick"])
        )
        render_time = (
            -1.0
            if self._last_frame_tick is None
            else self._last_frame_tick * float(clock["seconds_per_tick"])
        )
        lidar_time = (
            -1.0
            if self._last_scan_tick is None
            else self._last_scan_tick * float(clock["seconds_per_tick"])
        )
        current_pose = (
            *state["position_meters"],
            float(state["yaw_degrees"]),
            float(state["pitch_degrees"]),
            float(clock["elapsed_seconds"]),
        )
        if self._previous_pose is None:
            egomotion = np.zeros(6, dtype=np.float32)
        else:
            egomotion = np.asarray(current_pose, dtype=np.float32) - np.asarray(
                self._previous_pose, dtype=np.float32
            )
            # Headings are circular. Report the shortest signed rotation so
            # crossing 0/360 degrees cannot masquerade as a near-full turn.
            egomotion[3] = (egomotion[3] + 180.0) % 360.0 - 180.0
        self._previous_pose = current_pose
        center = tuple(size // 2 for size in voxels.shape)
        offsets = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
        local_relations = np.asarray(
            [
                bool(voxels[center[0] + dx, center[1] + dy, center[2] + dz] & 0xFFF)
                for dx, dy, dz in offsets
            ],
            dtype=np.int8,
        )
        dt = float(clock["seconds_per_tick"])
        return {
            "clock": np.asarray(
                [clock["tick"], clock["elapsed_seconds"], remaining_ticks, remaining_seconds, render_time, lidar_time],
                dtype=np.float64,
            ),
            "velocity": np.asarray(state["velocity_meters_per_second"], dtype=np.float32),
            "egomotion": egomotion,
            "spatial_meta": np.asarray(
                [state["frame_id"], state["scale"], state["meters_per_cell"]],
                dtype=np.float32,
            ),
            "sensor_age": np.asarray(
                [
                    -1.0 if self._last_frame_tick is None else (clock["tick"] - self._last_frame_tick) * dt,
                    -1.0 if self._last_scan_tick is None else (clock["tick"] - self._last_scan_tick) * dt,
                ],
                dtype=np.float32,
            ),
            "local_relations": local_relations,
        }

    def oracle_view(self) -> dict[str, Any]:
        """Return the recorder-only complete view, never part of policy input."""

        state = dict(self.world.oracle_state())
        nodes: list[dict[str, Any]] = [
            {
                "id": "agent:0",
                "kind": "agent",
                "attributes": {
                    "position_cells": state["position_cells"],
                    "position_meters": state["position_meters"],
                    "frame_id": state["frame_id"],
                },
            }
        ]
        relations: list[dict[str, Any]] = []
        reachability: list[dict[str, Any]] = []
        structure_nodes: set[str] = set()
        for entity in state.get("entities", ()):
            if entity.get("kind") == "agent":
                continue
            nodes.append(
                {
                    "id": f"{entity['kind']}:{entity['id']}",
                    "kind": entity["kind"],
                    "attributes": dict(entity),
                }
            )
        agent_cell = tuple(int(np.floor(value)) for value in state["position_cells"])
        query_world = self.world.fork()
        for region in state.get("semantic_regions", ()):
            region_id = f"region:{region['region_id']}"
            structure_id = f"structure:{region['structure_id']}"
            nodes.append(
                {"id": region_id, "kind": "region", "attributes": dict(region)}
            )
            if structure_id not in structure_nodes:
                structure_nodes.add(structure_id)
                nodes.append(
                    {"id": structure_id, "kind": "structure", "attributes": {}}
                )
            relations.append(
                {"kind": "defines", "source": region_id, "target": structure_id}
            )
            x0, y0, z0, x1, y1, z1 = region["bounds_cells"]
            if x0 <= agent_cell[0] <= x1 and y0 <= agent_cell[1] <= y1 and z0 <= agent_cell[2] <= z1:
                relations.append(
                    {"kind": "contains", "source": region_id, "target": "agent:0"}
                )
            target_cell = ((x0 + x1) // 2, y1 + 1, (z0 + z1) // 2)
            target_block = ((x0 + x1) // 2, (y0 + y1) // 2, (z0 + z1) // 2)
            try:
                path = query_world.shortest_path(
                    agent_cell, target_cell, max_visited=4096
                )
                reachable = path is not None
                reason = None
            except ValueError as exc:
                path = None
                reachable = None
                reason = str(exc)
            eye_position = (
                float(state["position_cells"][0]),
                float(state["position_cells"][1]) + 1.62 * float(state["scale"]),
                float(state["position_cells"][2]),
            )
            visible = query_world.visible(eye_position, target_block)
            reachability.append(
                {
                    "source": "agent:0",
                    "target": region_id,
                    "target_cell": target_cell,
                    "reachable": reachable,
                    "visible": bool(visible),
                    "shortest_path_cells": path,
                    "reason": reason,
                }
            )
            if reachable:
                relations.append(
                    {"kind": "reachable", "source": "agent:0", "target": region_id}
                )
            if visible:
                relations.append(
                    {"kind": "visible", "source": "agent:0", "target": region_id}
                )
        state.update(
            {
                "physics_config": deepcopy(self._physics),
                "physics": dict(self.world.physics()),
                "sensor_profile": self.sensor_profile(),
                "world_hash": self.world.hash(),
                "world_snapshot": bytes(self.world.snapshot()),
                "task_state": (
                    None
                    if self._task is None
                    else deepcopy(self._task.state_dict())
                ),
                "reward_outcome": self._last_reward.as_info(),
                "trace": deepcopy(self._last_trace),
                "events": (
                    []
                    if self._last_trace is None
                    else deepcopy(self._last_trace.get("events", []))
                ),
                "deltas": (
                    []
                    if self._last_trace is None
                    else deepcopy(self._last_trace.get("deltas", []))
                ),
                "terminated": self._terminated,
                "truncated": self._truncated,
                "intervention_cursor": self._intervention_cursor,
                "sensor_sample_ticks": {
                    "render": self._last_frame_tick,
                    "lidar": self._last_scan_tick,
                },
                "scene_graph": {"nodes": nodes, "relations": relations},
                "reachability": reachability,
                "spatial_query_capabilities": (
                    "nearest",
                    "within",
                    "adjacent",
                    "above",
                    "below",
                    "visible",
                    "reachable",
                    "shortest_path",
                    "connected_component",
                ),
            }
        )
        return state

    def sensor_profile(self) -> dict[str, Any]:
        """Serializable observation cadence and modality configuration."""

        return {
            "render_every": self._render_every,
            "lidar": deepcopy(self._lidar),
            "spacetime": self._spacetime,
            "modalities": sorted(self.observation_space.spaces),
        }

    def _scan(self):
        if self._last_scan is None or self.world.tick() % self._lidar_every == 0:
            cfg = self._lidar
            # frame_idx = current tick: noise is a pure function of
            # (world state, tick), so replayed episodes see identical scans
            self._last_scan = self.world.lidar_scan(
                channels=cfg["channels"],
                azimuth_steps=cfg["azimuth"],
                min_elev_deg=cfg.get("min_elev", -15.0),
                max_elev_deg=cfg.get("max_elev", 15.0),
                max_range=cfg.get("max_range", 64.0),
                noise_sigma=cfg.get("noise_sigma", 0.0),
                dropout_p=cfg.get("dropout_p", 0.0),
                noise_seed=cfg.get("noise_seed", 0),
                frame_idx=self.world.tick(),
            )
            self._last_scan_tick = self.world.tick()
        return self._last_scan

    def _frames(self):
        if self._last_frames is None or self.world.tick() % self._render_every == 0:
            rgb, depth, seg, normals = self.world.render()
            self._last_frames = (rgb, depth.astype(np.float16), seg, normals)
            self._last_frame_tick = self.world.tick()
        return self._last_frames


def _validate_snapshot_observation_profile(
    snapshot: EnvSnapshot,
    *,
    world_tick: int,
) -> tuple[int, dict[str, Any] | None, int, bool, spaces.Dict]:
    if (
        isinstance(snapshot.render_every, bool)
        or not isinstance(snapshot.render_every, int)
        or snapshot.render_every < 0
    ):
        raise ValueError("EnvSnapshot render_every must be a non-negative integer")
    render_every = snapshot.render_every
    lidar = deepcopy(snapshot.lidar_config)
    lidar_every = 0
    if lidar is not None:
        if not isinstance(lidar, dict):
            raise ValueError("EnvSnapshot lidar_config must be a mapping or null")
        for key in ("channels", "azimuth"):
            value = lidar.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"EnvSnapshot lidar_config {key!r} must be a positive integer"
                )
        every = lidar.get("every", 1)
        if isinstance(every, bool) or not isinstance(every, int) or every <= 0:
            raise ValueError(
                "EnvSnapshot lidar_config 'every' must be a positive integer"
            )
        lidar_every = every
    spacetime = bool(snapshot.spacetime)
    candidate_space = observation_space(render_every > 0, lidar, spacetime)

    for label, sample_tick in (
        ("render", snapshot.render_sample_tick),
        ("lidar", snapshot.lidar_sample_tick),
    ):
        if sample_tick is None:
            continue
        if (
            isinstance(sample_tick, bool)
            or not isinstance(sample_tick, int)
            or sample_tick < 0
            or sample_tick > world_tick
        ):
            raise ValueError(
                f"EnvSnapshot {label} sample tick must be within the restored clock"
            )
    if snapshot.last_frames is not None:
        expected = (
            ((RENDER_RES, RENDER_RES, 3), np.dtype(np.uint8)),
            ((RENDER_RES, RENDER_RES), np.dtype(np.float16)),
            ((RENDER_RES, RENDER_RES), np.dtype(np.uint16)),
            ((RENDER_RES, RENDER_RES, 3), np.dtype(np.float32)),
        )
        if len(snapshot.last_frames) != len(expected) or any(
            np.asarray(value).shape != shape or np.asarray(value).dtype != dtype
            for value, (shape, dtype) in zip(snapshot.last_frames, expected, strict=True)
        ):
            raise ValueError("EnvSnapshot render cache shape or dtype is invalid")
        if render_every == 0:
            raise ValueError("EnvSnapshot contains render cache while rendering is disabled")
    if snapshot.last_scan is not None:
        if lidar is None:
            raise ValueError("EnvSnapshot contains LiDAR cache without lidar_config")
        scan_shape = (lidar["channels"], lidar["azimuth"])
        expected_dtypes = (np.dtype(np.float32), np.dtype(np.float32), np.dtype(np.uint16))
        if len(snapshot.last_scan) != 3 or any(
            np.asarray(value).shape != scan_shape or np.asarray(value).dtype != dtype
            for value, dtype in zip(snapshot.last_scan, expected_dtypes, strict=True)
        ):
            raise ValueError("EnvSnapshot LiDAR cache shape or dtype is invalid")
    if snapshot.previous_pose is not None and (
        len(snapshot.previous_pose) != 6
        or not all(np.isfinite(value) for value in snapshot.previous_pose)
    ):
        raise ValueError("EnvSnapshot previous_pose must contain six finite values")
    if snapshot.last_trace is not None and not isinstance(snapshot.last_trace, dict):
        raise ValueError("EnvSnapshot last_trace must be a mapping or null")
    return render_every, lidar, lidar_every, spacetime, candidate_space


def _capture_native_trace_state(world) -> bytes | None:
    """Capture optional native trace bookkeeping without requiring new bindings."""

    for name in ("trace_state_snapshot", "snapshot_trace_state"):
        capture = getattr(world, name, None)
        if callable(capture):
            value = capture()
            return None if value is None else bytes(value)
    return None


def _validate_task_reward_contract(task: Any) -> None:
    if task is None:
        return
    if not isinstance(task, StatefulTask):
        raise TypeError(
            "VoxelGymEnv tasks must implement serializable "
            "state_dict/load_state_dict"
        )
    validate = getattr(task, "validate_reward_contract", None)
    if callable(validate):
        validate()
    reward_outcome = getattr(task, "reward_outcome", None)
    if not callable(reward_outcome):
        raise TypeError(
            "VoxelGymEnv tasks must implement pure reward_outcome(world, events=()) "
            "returning RewardOutcome"
        )


def _task_state_fingerprint(state: dict[str, Any]) -> str:
    try:
        return json.dumps(
            state,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("task state_dict() must return JSON-safe state") from exc


def _invoke_reward_outcome(callback, world, events: tuple[dict[str, Any], ...]):
    """Pass causal context to the v1 reward contract with one-arg fallback."""

    parameters = tuple(inspect.signature(callback).parameters.values())
    accepts_events = len(parameters) >= 2 or any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in parameters
    )
    return callback(world, events) if accepts_events else callback(world)


def _call_with_read_only_world(world, callback, label: str):
    """Evaluate a task callback on an isolated world and reject mutation."""

    probe = world.fork()
    before = (
        bytes(probe.snapshot()),
        _capture_native_trace_state(probe),
        _capture_native_intervention_cursor(probe),
    )
    result = callback(probe)
    after = (
        bytes(probe.snapshot()),
        _capture_native_trace_state(probe),
        _capture_native_intervention_cursor(probe),
    )
    if after != before:
        raise TypeError(
            f"task {label}(world) mutated its read-only World; return a "
            "serializable intervention or RewardOutcome instead"
        )
    return result


def _restore_native_trace_state(world, payload: bytes | None) -> None:
    if payload is None:
        return
    restore = getattr(world, "restore_trace_state", None)
    if not callable(restore):
        raise ValueError(
            "snapshot contains native trace state but this runtime cannot restore it"
        )
    restore(payload)


def _capture_native_intervention_cursor(world) -> int:
    capture = getattr(world, "intervention_cursor", None)
    return 0 if not callable(capture) else int(capture())


def _restore_native_intervention_cursor(world, cursor: int) -> None:
    restore = getattr(world, "restore_intervention_cursor", None)
    if not callable(restore):
        if cursor:
            raise ValueError(
                "snapshot contains a native intervention cursor but this runtime "
                "cannot restore it"
            )
        return
    restore(int(cursor))
