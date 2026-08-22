"""Probe stage-0 re-harvest behavior."""
import math
import sys
sys.path.insert(0, "python")
from voxelgym import ids
from voxelgym.env import VoxelGymEnv
from voxelgym.tasks import make_task
from voxelgym.experts import make_expert

task = make_task("craft_stone_pickaxe")
env = VoxelGymEnv(task=task, seed=0)
env.reset(seed=0)
ex = make_expert("craft_stone_pickaxe", task, seed=0)

for i in range(165):
    env.step(ex.act(env.world))

op = ex.stages[0].op
for i in range(30):
    a = ex.act(env.world)
    env.step(a)
    x, y, z = env.world.agent_pos()
    ch = env.world.crosshair()
    drops = env.world.drops_of(ids.LOG)
    print(f"t={env.world.tick()} pos=({x:.2f},{y:.2f},{z:.2f}) mv={a['move']},m={a['mine']} yaw={a['yaw']} "
          f"focus={op.focus} tgt={op.target} drops={[(round(dx,1),round(dy,1),round(dz,1)) for dx,dy,dz in drops]} "
          f"nodrop={op.no_drops_until} expl={op.explore_left}", flush=True)
