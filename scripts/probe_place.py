"""Dump geometry at seed-0 place_furnace stall."""
import sys
sys.path.insert(0, "python")
from voxelgym import ids
from voxelgym.env import VoxelGymEnv
from voxelgym.tasks import make_task
from voxelgym.experts import make_expert

task = make_task("smelt_iron")
env = VoxelGymEnv(task=task, seed=0)
env.reset(seed=0)
ex = make_expert("smelt_iron", task, seed=0)
for i in range(12000):
    env.step(ex.act(env.world))
    if ex._idx >= 13:
        break
print("stage", ex._idx, "tick", env.world.tick())
# run to stall point: already there since place never completes
x, y, z = env.world.agent_pos()
print("pos", x, y, z, "on_ground", env.world.obs_pose()[5])
names = {0: ".", 1: "B", 2: "S", 3: "D", 4: "G", 5: "s", 6: "g", 7: "~", 9: "T", 10: "l", 11: "P", 12: "C", 13: "F", 19: "c"}
ax, ay, az = int(x), int(y), int(z)
for yy in range(ay + 2, ay - 3, -1):
    for zz in range(az - 2, az + 3):
        row = ""
        for xx in range(ax - 2, ax + 3):
            c = env.world.get_block(xx, yy, zz) & 0xFFF
            row += names.get(c, "?") if (xx, yy, zz) != (ax, ay, az) else "@"
        print(f"y={yy:3d} z={zz:3d} {row}")
# try every combo: print resulting crosshair
for pitch in (7, 8, 6):
    for yaw in (18, 12, 0, 6):
        env.world.step((0, 0, 0, yaw, pitch, 0, 0, 0, 0, 0))
        print("yaw", yaw, "pitch", pitch, "->", env.world.crosshair())
