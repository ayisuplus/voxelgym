"""Dump geometry at seed-0 place_furnace stall."""
from _harness import boot, run_until, dump_slice

task, env, ex = boot("smelt_iron", seed=0)
run_until(env, ex, 12000, stop=lambda e, x: x._idx >= 13)
print("stage", ex._idx, "tick", env.world.tick())
# run to stall point: already there since place never completes
x, y, z = env.world.agent_pos()
print("pos", x, y, z, "on_ground", env.world.obs_pose()[5])
dump_slice(env.world)
# try every combo: print resulting crosshair
for pitch in (7, 8, 6):
    for yaw in (18, 12, 0, 6):
        env.world.step((0, 0, 0, yaw, pitch, 0, 0, 0, 0, 0))
        print("yaw", yaw, "pitch", pitch, "->", env.world.crosshair())
