"""Probe stage-0 re-harvest behavior."""
from _harness import boot, run_until
from voxelgym import ids

task, env, ex = boot("craft_stone_pickaxe", seed=0)
run_until(env, ex, 165)

op = ex.stages[0].op
for i in range(30):
    a = ex.act(env.world)
    env.step(a)
    x, y, z = env.world.agent_pos()
    drops = env.world.drops_of(ids.LOG)
    print(f"t={env.world.tick()} pos=({x:.2f},{y:.2f},{z:.2f}) mv={a['move']},m={a['mine']} yaw={a['yaw']} "
          f"focus={op.focus} tgt={op.target} drops={[(round(dx,1),round(dy,1),round(dz,1)) for dx,dy,dz in drops]} "
          f"nodrop={op.no_drops_until} expl={op.explore_left}", flush=True)
