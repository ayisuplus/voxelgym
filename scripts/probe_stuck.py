"""Per-tick action+state dump for the frozen harvest situation."""
from _harness import boot, run_until

task, env, ex = boot("craft_stone_pickaxe", seed=0)
run_until(env, ex, 1490)

for i in range(40):
    op = ex.stages[6].op
    a = ex.act(env.world)
    env.step(a)
    x, y, z = env.world.agent_pos()
    print(f"t={env.world.tick()} pos=({x:.2f},{y:.2f},{z:.2f}) a={a} "
          f"focus={op.focus} desc={op.desc} walk={op.walk} vel? ray={tuple(env.world.obs_raycast())}", flush=True)
