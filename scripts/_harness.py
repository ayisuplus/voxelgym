"""Shared bootstrap for scripts/*.py debug drivers.

One env+expert factory, a fast-forward helper, an inventory formatter, and
one ASCII voxel-slice dump (the two local copies had already drifted apart:
trace_expert's name map was missing 19:"c").

Run scripts from the repo root:  python scripts/probe_place.py
"""
import sys

sys.path.insert(0, "python")

from voxelgym.env import VoxelGymEnv  # noqa: E402
from voxelgym.tasks import make_task  # noqa: E402
from voxelgym.experts import make_expert  # noqa: E402

# block id -> char for dump_slice (registry ids, see voxel-core block.rs)
SLICE_NAMES = {0: ".", 1: "B", 2: "S", 3: "D", 4: "G", 5: "s", 6: "g",
               7: "~", 8: "L", 9: "T", 10: "l", 11: "P", 12: "C", 13: "F", 19: "c"}


def boot(task_name: str, seed: int = 0):
    """(task, env, expert) after reset — the former 6-line script preamble."""
    task = make_task(task_name)
    env = VoxelGymEnv(task=task, seed=seed)
    env.reset(seed=seed)
    return task, env, make_expert(task_name, task, seed=seed)


def run_until(env, ex, cap: int, stop=None):
    """Step the expert up to cap ticks; stop(env, ex) -> true breaks early."""
    for _ in range(cap):
        env.step(ex.act(env.world))
        if stop is not None and stop(env, ex):
            return


def inv_slots(world):
    """Non-empty inventory slots as (item, count) pairs."""
    return [(int(s[0]), int(s[1])) for s in world.obs_inventory() if s[1] > 0]


def dump_slice(world, xr=(-2, 2), zr=(-2, 2), above=2, below=2, names=None):
    """ASCII dump of the voxel neighborhood around the agent (@ = agent)."""
    names = names or SLICE_NAMES
    x, y, z = world.agent_pos()
    ax, ay, az = int(x), int(y), int(z)
    for yy in range(ay + above, ay - below - 1, -1):
        for zz in range(az + zr[0], az + zr[1] + 1):
            row = ""
            for xx in range(ax + xr[0], ax + xr[1] + 1):
                c = world.get_block(xx, yy, zz) & 0xFFF
                row += names.get(c, "?") if (xx, yy, zz) != (ax, ay, az) else "@"
            print(f"y={yy:3d} z={zz:3d} {row}")
