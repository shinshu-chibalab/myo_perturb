import mujoco
# import myo_sim
import numpy as np
from mujoco import MjModel, MjData, mj_step
import imageio.v2 as imageio
import os

model = MjModel.from_xml_path("myo_sim/models/leg/myolegs32.xml")
data = MjData(model)
mujoco.mj_forward(model, data)
print("dt =", model.opt.timestep)
# ==========================================
# subtree COM をすべて表示
# ==========================================

print("subtree_com shape:", data.subtree_com.shape)
print("nbody:", model.nbody)

print(
    f"{'ID':>3} "
    f"{'Body':<30} "
    f"{'Body mass [kg]':>15} "
    f"{'subtree COM [x, y, z]'}"
)

print("-" * 90)

for body_id in range(model.nbody):

    body_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id
    )

    body_mass = model.body_mass[body_id]

    com = data.subtree_com[body_id]

    print(
        f"{body_id:3d} "
        f"{str(body_name):<30} "
        f"{body_mass:15.6f} "
        f"[{com[0]: .6f}, {com[1]: .6f}, {com[2]: .6f}]"
    )


# ==========================================
# world と Full Body の COM を比較
# ==========================================

full_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "Full Body"
)

print()
print("world body id     :", 0)
print("Full Body body id :", full_body_id)

print(
    "world subtree COM     :",
    data.subtree_com[0]
)

print(
    "Full Body subtree COM :",
    data.subtree_com[full_body_id]
)


# ==========================================
# 筋名を表示
# ==========================================

muscle_names = []

renderer = mujoco.Renderer(model, height=400, width=400)
frames = []

for i in range(model.nu):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )

    # ==========================================
    # 初期 MTU / actuator length
    # ==========================================

    l_mtu = data.actuator_length[i]

    print(l_mtu)

def get_tendon_length(model, actuator_id):

    r_min = model.actuator_gainprm[actuator_id, 0]
    r_max = model.actuator_gainprm[actuator_id, 1]

    l_min = model.actuator_lengthrange[actuator_id, 0]
    l_max = model.actuator_lengthrange[actuator_id, 1]

    L0 = (l_max - l_min) / (r_max - r_min)

    LT = l_min - r_min * L0

    return LT

tendon_length = {}

for i in range(model.nu):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )

    tendon_length[name] = get_tendon_length(
        model,
        i
    )

for i in range(model.nu):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )

    # muscle "range"
    r_min = model.actuator_gainprm[i, 0]
    r_max = model.actuator_gainprm[i, 1]

    # actuator transmission length range
    length_min = model.actuator_lengthrange[i, 0]
    length_max = model.actuator_lengthrange[i, 1]

    # MuJoCo optimal muscle length L0
    l_opt = (
        (length_max - length_min)
        / (r_max - r_min)
    )

    # MuJoCo constant tendon-length equivalent LT
    l_tendon = length_min - r_min * l_opt

    # MTU length when normalized muscle length L = 1
    l_mtu_opt = l_tendon + l_opt

    print(
        name,
        f"L0={l_opt:.6f}",
        f"LT={l_tendon:.6f}",
        f"L_MTU_opt={l_mtu_opt:.6f}"
    )

for i in range(model.nactuator):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )

    muscle_names.append(name)

    print(i, name)


# ==========================================
# 各筋への入力
# ==========================================

u = np.zeros(model.nu)

# 例：筋ごとに適当な値

# 全筋へ入力
data.ctrl[:] = u

print("ncam =", model.ncam)

for i in range(model.ncam):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i))

# ==========================================
# simulation
# ==========================================


for step in range(1000):

    data.ctrl[:] = u
    # if step % 100 == 0:
    #     print(data.subtree_com[1])

    mujoco.mj_step(
        model,
        data
    )

    renderer.update_scene(data, camera="front_view")
    frames.append(renderer.render())

os.makedirs("videos", exist_ok=True)

with imageio.get_writer("videos/myolegs32_simulation.mp4", fps=200, codec="libx264", pixelformat="yuv420p") as writer:
    for frame in frames:
        writer.append_data(frame)