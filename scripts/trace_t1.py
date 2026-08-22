"""Trace T1 expert on a failing seed."""
import math
import sys
sys.path.insert(0, "python")
from voxelgym.env import VoxelGymEnv
from voxelgym.tasks import make_task
from voxelgym.experts import make_expert

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 4
task = make_task("navigate_to_target")
env = VoxelGymEnv(task=task, seed=seed)
env.reset(seed=seed)
ex = make_expert("navigate_to_target", task, seed=seed)
print("target", task.target)

last = None
for i in range(2400):
    a = ex.act(env.world)
    env.step(a)
    x, y, z = env.world.agent_pos()
    d = math.hypot(x - task.target[0], z - task.target[2])
    if i % 100 == 0:
        print(f"t={i} pos=({x:.1f},{y:.1f},{z:.1f}) d={d:.1f} a={a['move']},{a['jump']} yaw={a['yaw']} hp={env.world.hp()} stall={ex.nav._stall} detour={ex.nav._detour}", flush=True)
    if env.world.dead():
        print("DIED at", i, "pos", env.world.agent_pos())
        break
    if d <= 2.0:
        print("SUCCESS at", i)
        break
else:
    print("horizon, final d=", d, "pos", env.world.agent_pos())
