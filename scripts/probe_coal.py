"""Probe the coal harvest stall on a seed."""
import sys

from _harness import boot, run_until

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
task, env, ex = boot("smelt_iron", seed=seed)
run_until(env, ex, 12000, stop=lambda e, x: x._idx >= 14 and e.world.tick() > 11500)
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
