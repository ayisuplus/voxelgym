"""gymnasium environment over the Rust voxel core.

Observation/action spaces follow the plan contract literally.

Cell encoding in `voxels`: raw u16 cell = (state << 12) | block_id.
  - low 12 bits: block id (registry truth table)
  - high 4 bits: state (fluid level 0..15, wire power 0..15, door/lever bit)

`raycast` distance is in centi-cells (450 = 4.5-cell reach cap, no target).
"""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import voxelgym_rs as rs

ACTION_KEYS = ("move", "jump", "sneak", "yaw", "pitch", "mine", "place", "use", "hotbar", "craft")

RENDER_RES = 128


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


def observation_space(render: bool = False, lidar: dict | None = None) -> spaces.Dict:
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
    return spaces.Dict(d)


class VoxelGymEnv(gym.Env):
    """Single voxel world. `task` is a duck-typed Task (M2/M3): it provides
    preset/horizon/scenario(rng)/on_reset(world)/step_reward(world) and this
    env handles the sim + obs plumbing.

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
    ):
        super().__init__()
        self._task = task
        self._preset = preset or getattr(task, "preset", None) or "default"
        self._seed0 = seed
        self._scale = float(scale)
        if render is True:
            render = 1
        self._render_every = int(render) if render else 0
        self._lidar = dict(lidar) if lidar else None
        self._lidar_every = int(self._lidar.get("every", 1)) if self._lidar else 0
        self._last_scan: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.action_space = action_space()
        self.observation_space = observation_space(self._render_every > 0, self._lidar)
        self._w: rs.PyWorld | None = None
        self._episode_seed = seed
        self._last_frames: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    @property
    def world(self) -> rs.PyWorld:
        assert self._w is not None, "call reset() first"
        return self._w

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._episode_seed = self._seed0 if seed is None else seed
        scenario = None
        if self._task is not None:
            scenario = self._task.scenario(self.np_random)
        self._w = rs.PyWorld(self._episode_seed, self._preset, scenario, scale=self._scale)
        self._last_frames = None
        self._last_scan = None
        if self._task is not None:
            self._task.on_reset(self._w, self.np_random)
        return self._obs(), {"seed": self._episode_seed}

    def step(self, action: dict[str, Any]):
        w = self.world
        w.step(tuple(int(action[k]) for k in ACTION_KEYS))
        reward = 0.0
        terminated = bool(w.dead())
        if self._task is not None:
            r, t = self._task.step_reward(w)
            reward += r
            terminated = terminated or t
        truncated = False
        horizon = getattr(self._task, "horizon", None)
        if horizon is not None and w.tick() >= horizon:
            truncated = True
        return self._obs(), reward, terminated, truncated, {"tick": w.tick()}

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
        return obs

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
        return self._last_scan

    def _frames(self):
        if self._last_frames is None or self.world.tick() % self._render_every == 0:
            rgb, depth, seg, normals = self.world.render()
            self._last_frames = (rgb, depth.astype(np.float16), seg, normals)
        return self._last_frames
