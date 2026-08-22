"""Watch the cobble stage across its whole life on a seed."""
import math
import sys
sys.path.insert(0, "python")
from voxelgym import ids
from voxelgym.env import VoxelGymEnv
from voxelgym.tasks import make_task
from voxelgym.experts import make_expert

seed = int(sys.argv[1])
task = make_task("craft_stone_pickaxe")
env = VoxelGymEnv(task=task, seed=seed)
env.reset(seed=seed)
ex = make_expert("craft_stone_pickaxe", task, seed=seed)
# fast forward to stage 7 (cobble)
for i in range(12000):
    env.step(ex.act(env.world))
    if ex._idx >= 7:
        break
print("cobble stage at tick", env.world.tick(), flush=True)
op = ex.stages[7].op
for i in range(6000):
    a = ex.act(env.world)
    env.step(a)
    if i % 150 == 0:
        x, y, z = env.world.agent_pos()
        cobble = env.world.count_item(ids.COBBLESTONE)
        print(f"t={env.world.tick()} pos=({x:.0f},{y:.0f},{z:.0f}) tgt={op.target} desc={op.desc is not None} "
              f"focus={op.focus} recover={op._recover_until > env.world.tick()} lastgain={env.world.tick()-op._last_gain} cobble={cobble}", flush=True)
    if env.world.count_item(ids.COBBLESTONE) >= 12:
        print("DONE at", env.world.tick())
        break
