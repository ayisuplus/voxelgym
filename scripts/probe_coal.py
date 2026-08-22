"""Probe the coal harvest stall on a seed."""
import math
import sys
sys.path.insert(0, "python")
from voxelgym import ids
from voxelgym.env import VoxelGymEnv
from voxelgym.tasks import make_task
from voxelgym.experts import make_expert

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
task = make_task("smelt_iron")
env = VoxelGymEnv(task=task, seed=seed)
env.reset(seed=0)
ex = make_expert("smelt_iron", task, seed=seed)
for i in range(12000):
    env.step(ex.act(env.world))
    if ex._idx >= 14 and env.world.tick() > 11500:
        break
print("stage", ex._idx, "tick", env.world.tick(), flush=True)
op = ex.stages[min(ex._idx, len(ex.stages) - 1)].op
if not hasattr(op, "focus"):
    op = ex.stages[14].op
for i in range(60):
    a = ex.act(env.world)
    env.step(a)
    x, y, z = env.world.agent_pos()
    print(f"t={env.world.tick()} pos=({x:.2f},{y:.2f},{z:.2f}) mv={a['move']},j={a['jump']},m={a['mine']} yaw={a['yaw']} p={a['pitch']} "
          f"focus={op.focus} tgt={op.target} desc={op.desc} stall={op._stall}", flush=True)
