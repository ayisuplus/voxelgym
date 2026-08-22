"""Debug: trace GatherExpert stage progression for one episode."""
import sys, time
sys.path.insert(0, "python")
from voxelgym import ids
from voxelgym.env import VoxelGymEnv
from voxelgym.tasks import make_task
from voxelgym.experts import make_expert

task = make_task("craft_stone_pickaxe")
env = VoxelGymEnv(task=task, seed=0)
env.reset(seed=0)
ex = make_expert("craft_stone_pickaxe", task, seed=0)

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
        inv = env.world.obs_inventory()
        slots = [(i, int(s[0]), int(s[1])) for i, s in enumerate(inv) if s[1] > 0]
        print(f"t={env.world.tick():5d} stage->{idx} pos=({x:.1f},{y:.1f},{z:.1f}) slots={slots}", flush=True)
        last_stage = idx
    a = ex.act(env.world)
    obs, r, term, trunc, _ = env.step(a)
    if term or trunc:
        print("END", term, trunc, "dead=", env.world.dead())
        break
    if i % 1000 == 999:
        x, y, z = env.world.agent_pos()
        inv = env.world.obs_inventory()
        slots = [(int(s[0]), int(s[1])) for s in inv if s[1] > 0]
        print(f"  ...t={env.world.tick()} stage={idx} pos=({x:.1f},{y:.1f},{z:.1f}) slots={slots} elapsed={time.time()-t0:.1f}s", flush=True)
    if idx == 6 and i % 500 == 0:
        op = ex.stages[6].op
        ray = env.world.obs_raycast()
        fc = op.focus
        print(f"    t={env.world.tick()} focus={fc} ray=({ray[0]},{ray[1]}) target={op.target} desc={op.desc} stall={op._stall}", flush=True)
        ax, ay, az = env.world.agent_pos()
        ax, ay, az = int(ax), int(ay), int(az)
        names = {0: ".", 1: "B", 2: "S", 3: "D", 4: "G", 5: "s", 6: "g", 7: "~", 8: "L", 9: "T", 10: "l", 11: "P", 12: "C", 13: "F"}
        for yy in range(ay + 2, ay - 3, -1):
            for zz in range(az - 1, az + 4):
                row = ""
                for xx in range(ax - 2, ax + 4):
                    c = env.world.get_block(xx, yy, zz) & 0xFFF
                    row += names.get(c, "?") if (xx, yy, zz) != (ax, ay, az) else "@"
                print(f"      y={yy:3d} z={zz:3d} {row}", flush=True)
print("done", time.time() - t0)
