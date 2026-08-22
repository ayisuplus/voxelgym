"""Per-tick action+state dump for the frozen harvest situation."""
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

# fast-forward to the stuck region
for i in range(1490):
    a = ex.act(env.world)
    env.step(a)

for i in range(40):
    op = ex.stages[6].op
    a = ex.act(env.world)
    env.step(a)
    x, y, z = env.world.agent_pos()
    print(f"t={env.world.tick()} pos=({x:.2f},{y:.2f},{z:.2f}) a={a} "
          f"focus={op.focus} desc={op.desc} walk={op.walk} vel? ray={tuple(env.world.obs_raycast())}", flush=True)
