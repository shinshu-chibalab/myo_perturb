import mujoco
import numpy as np
from mujoco import MjModel, MjData, mj_step

model = MjModel.from_xml_path("myo_sim/models/leg/myolegs32.xml")
data = MjData(model)

print("njnt =", model.njnt)
print("nv   =", model.nv)
print("nu   =", model.nu)

print("\nJoints")
for i in range(model.njnt):
    print(
        i,
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            i
        )
    )

print("\nActuators")
for i in range(model.nu):
    print(
        i,
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            i
        )
    )

mujoco.mj_forward(model, data)


def get_actuator_moment_arm(data, actuator_id, dof_id):
    """
    actuator_id の dof_id に対する moment arm を取得する。
    MuJoCo の sparse actuator_moment に対応。
    """

    row_adr = data.moment_rowadr[actuator_id]
    row_nnz = data.moment_rownnz[actuator_id]

    # この actuator が作用するDOF番号
    cols = data.moment_colind[
        row_adr:row_adr + row_nnz
    ]

    # それぞれのDOFに対するmoment arm
    vals = data.actuator_moment[
        row_adr:row_adr + row_nnz
    ]

    # 欲しいDOFを探す
    idx = np.where(cols == dof_id)[0]

    if len(idx) == 0:
        return 0.0

    return vals[idx[0]]


lumbar_joints = [
    "flex_extension",
    "lat_bending",
    "axial_rotation",
]

lumbar_muscles = [
    "ercspn_r",
    "ercspn_l",
    "intobl_r",
    "intobl_l",
    "extobl_r",
    "extobl_l",
]

print("=== Lumbar muscle moment arms ===")

for muscle_name in lumbar_muscles:

    aid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        muscle_name
    )

    print(f"\n{muscle_name}")

    for joint_name in lumbar_joints:

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )

        dof_id = model.jnt_dofadr[jid]

        moment_arm = get_actuator_moment_arm(
            data,
            aid,
            dof_id
        )

        print(
            f"  {joint_name:20s}: "
            f"{moment_arm:+.6f}"
        )