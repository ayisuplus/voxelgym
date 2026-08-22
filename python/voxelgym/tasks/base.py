"""Task base protocol. Tasks are duck-typed into VoxelGymEnv:

- preset: world preset used to construct the world
- horizon: tick budget (truncation)
- scenario(rng) -> scenario spec list | None (deterministic in episode seed)
- on_reset(world, rng): place markers, teleport, grant items
- step_reward(world) -> (reward, terminated)
"""

from __future__ import annotations

import numpy as np


class Task:
    name = "base"
    preset = "default"
    horizon: int | None = None

    def scenario(self, rng: np.random.Generator):
        return None

    def on_reset(self, world, rng: np.random.Generator):
        pass

    def step_reward(self, world) -> tuple[float, bool]:
        return 0.0, False
