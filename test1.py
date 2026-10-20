import numpy as np
from x0_myolegs32 import *
import mujoco
from mujoco import MjModel, MjData, mj_step, mj_forward

muscles = myolegs32_v0

model = MjModel.from_xml_path("myo_sim\models\leg\myolegs32.xml")
actuator_ids = [model.actuator(m).id for m in muscles]

print(actuator_ids)

tendon_ids = np.empty(len(actuator_ids), dtype=int)

for i, aid in enumerate(actuator_ids):

    # tendonを介したactuatorか確認
    if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_TENDON:
        raise ValueError(
            f"Actuator '{model.actuator(aid).name}' "
            f"is not a tendon-transmission actuator."
        )

    # actuatorが接続されているtendon ID
    tendon_ids[i] = model.actuator_trnid[aid, 0]

print(tendon_ids)