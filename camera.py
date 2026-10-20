import mujoco
import mujoco.viewer

# model = mujoco.MjModel.from_xml_path("myo_sim/gait10dof18musc/gait10dof18musc_cvt5.xml")
# model = mujoco.MjModel.from_xml_path("myo_sim/gait2354/gait2354_simbody_cvt3.xml")
# model = mujoco.MjModel.from_xml_path("myo_sim/torso/myotorso_abdomen.xml")
model = mujoco.MjModel.from_xml_path("myo_sim\models\leg\myolegs32.xml")
data = mujoco.MjData(model)

# キーフレーム初期化
data.qpos[:] = model.qpos0
data.qvel[:] = 0
data.qacc[:] = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
