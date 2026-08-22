"""Report final stage per seed for a T2 task."""
import sys
sys.path.insert(0, "python")
from voxelgym import ids
from voxelgym.env import VoxelGymEnv
from voxelgym.tasks import make_task
from voxelgym.experts import make_expert

NAMES = ["log", "planks", "stick", "table", "place_table", "wpick", "cobble", "return_table", "spick",
         "furnace", "place_furnace", "coal", "iron_ore", "smelt", "return2", "ipick", "diamond"]

task_name = sys.argv[1] if len(sys.argv) > 1 else "craft_stone_pickaxe"
seeds = [int(s) for s in sys.argv[2:]] or list(range(20))
import os
cap = int(os.environ.get("CAP_TICKS", "12000"))

for seed in seeds:
    task = make_task(task_name)
    env = VoxelGymEnv(task=task, seed=seed)
    env.reset(seed=seed)
    ex = make_expert(task_name, task, seed=seed)
    end_tick = 0
    marks = []
    last_idx = 0
    for i in range(cap):
        a = ex.act(env.world)
        _, _, term, trunc, _ = env.step(a)
        end_tick = i
        if getattr(ex, "_idx", 0) != last_idx:
            marks.append((last_idx, ex._idx, env.world.tick()))
            last_idx = ex._idx
        if term or trunc:
            break
    print(f"  seed={seed} transitions: {marks}", flush=True)
    idx = getattr(ex, "_idx", 0)
    opname = type(ex.stages[min(idx, len(ex.stages) - 1)].op).__name__ if hasattr(ex, "stages") and idx < len(ex.stages) else "done"
    inv = env.world.obs_inventory()
    slots = [(int(s[0]), int(s[1])) for s in inv if s[1] > 0]
    x, y, z = env.world.agent_pos()
    status = "OK" if term and not env.world.dead() else f"stuck@{idx}:{opname}"
    print(f"seed={seed}: {status} t={end_tick} pos=({x:.0f},{y:.0f},{z:.0f}) slots={slots}", flush=True)
