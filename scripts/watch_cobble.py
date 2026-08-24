"""Watch the cobble stage across its whole life on a seed."""
import sys

from _harness import boot, run_until
from voxelgym import ids

seed = int(sys.argv[1])
task, env, ex = boot("craft_stone_pickaxe", seed=seed)
run_until(env, ex, 12000, stop=lambda e, x: x._idx >= 7)
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
