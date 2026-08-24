"""Debug: trace GatherExpert stage progression for one episode."""
import time

from _harness import boot, inv_slots, dump_slice

task, env, ex = boot("craft_stone_pickaxe", seed=0)

last_stage = None
t0 = time.time()
for i in range(12000):
    # find current stage index
    idx = None
    for j, st in enumerate(ex.stages):
        if not st.guard(env.world):
            idx = j
            break
    if idx != last_stage:
        x, y, z = env.world.agent_pos()
        slots = [(j, it, c) for j, (it, c) in enumerate(inv_slots(env.world))]
        print(f"t={env.world.tick():5d} stage->{idx} pos=({x:.1f},{y:.1f},{z:.1f}) slots={slots}", flush=True)
        last_stage = idx
    a = ex.act(env.world)
    obs, r, term, trunc, _ = env.step(a)
    if term or trunc:
        print("END", term, trunc, "dead=", env.world.dead())
        break
    if i % 1000 == 999:
        x, y, z = env.world.agent_pos()
        print(f"  ...t={env.world.tick()} stage={idx} pos=({x:.1f},{y:.1f},{z:.1f}) "
              f"slots={inv_slots(env.world)} elapsed={time.time()-t0:.1f}s", flush=True)
    if idx == 6 and i % 500 == 0:
        op = ex.stages[6].op
        ray = env.world.obs_raycast()
        print(f"    t={env.world.tick()} focus={op.focus} ray=({ray[0]},{ray[1]}) "
              f"target={op.target} desc={op.desc} stall={op._stall}", flush=True)
        dump_slice(env.world, xr=(-2, 3), zr=(-1, 3))
print("done", time.time() - t0)
