import mujoco
from mujoco import MjModel, MjData, mj_step, mj_forward
import numpy as np
from utils.param_utils import expand_params
from utils.metabolism import Wang2012

_WORKER = {}

def _init_worker(model_path, muscles, sim_steps, model_key_qpos, com_init_height, symmetry):

    if "myolegs26" in model_path:
        from utils.muscle_data_myolegs26 import get_parameter_array
    elif "myolegs32" in model_path:
        from utils.muscle_data_myolegs32 import get_parameter_array
    else:
        raise ValueError(
            f"Unsupported model_path: {model_path}. "
            "Expected model_path containing 'gait10dof18musc' or 'gait2354' or 'gait20dof24musc."
        )

    model = MjModel.from_xml_path(model_path)
    data = MjData(model)

    actuator_ids = np.array([model.actuator(m).id for m in muscles], dtype=int)

    # store frequently used constants in global var for worker
    _WORKER['model'] = model
    _WORKER['data'] = data
    _WORKER['actuator_ids'] = actuator_ids
    _WORKER['muscles'] = muscles
    _WORKER['symmetry'] = symmetry
    _WORKER['sim_steps'] = sim_steps
    _WORKER['model_qpos0'] = model_key_qpos.copy()
    _WORKER['com_init_height'] = com_init_height
    _WORKER['pelvis_tilt_id'] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_tilt")
    _WORKER['ankle_angle_r_id'] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_angle_r")
    _WORKER['ankle_angle_l_id'] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_angle_l")
    _WORKER['muscle_mass'] = get_parameter_array(list(muscles.keys()), "muscle_mass")
    _WORKER['l_opt'] = get_parameter_array(list(muscles.keys()), "l_opt")
    _WORKER['l_t'] = get_parameter_array(list(muscles.keys()), "l_t")
    _WORKER['slow_twitch_ratio'] = get_parameter_array(list(muscles.keys()), "slow_twitch_ratio")


def run_simulation_worker(x, controller, evaluator, delay_time=0.0, noise_std=0.0,):
    """
    この関数は各ワーカー内で実行される（_WORKER を参照）
    compressed_params: 圧縮（右側のみ）パラメータ
    """
    model = _WORKER['model']
    data = _WORKER['data']
    actuator_ids = _WORKER['actuator_ids']
    sim_steps = _WORKER['sim_steps']
    muscles = _WORKER['muscles']
    symmetry = _WORKER['symmetry']
    qpos0 = _WORKER['model_qpos0']
    com_init_height = _WORKER['com_init_height']
    muscle_mass = _WORKER['muscle_mass']
    l_opt = _WORKER['l_opt']
    l_t = _WORKER['l_t']
    slow_twitch_ratio = _WORKER['slow_twitch_ratio']
    # pelvis_tilt_id = _WORKER['pelvis_tilt_id']
    # ankle_angle_r_id = _WORKER['ankle_angle_r_id']
    # ankle_angle_l_id = _WORKER['ankle_angle_l_id']

    values = expand_params(x, muscles, symmetry=symmetry)

    n_muscles = len(muscles)
    zero = np.zeros(n_muscles)

    Kp = values.get("Kp", zero)
    l0 = values.get("l0", zero)
    Kd = values.get("Kd", zero)
    v0 = values.get("v0", zero)
    Kf = values.get("Kf", zero)
    f0 = values.get("f0", zero)
    ff = values.get("ff", zero)

    dt = model.opt.timestep
    delay_steps = max(0, int(delay_time / dt))
    buffer_size = delay_steps + 1

    all_costs = []

    run_count = 1 if noise_std <= 0 else 3

    for _ in range(run_count):
    
        length_buffer = np.zeros((buffer_size, n_muscles))
        velocity_buffer = np.zeros((buffer_size, n_muscles))
        force_buffer = np.zeros((buffer_size, n_muscles))

        com_log = np.zeros((sim_steps, 3))

        total_Edot = 0
        fall_cost = 0.0
        fallen = False

        mujoco.mj_resetData(model, data)
        
        data.qpos[:] = qpos0
        data.qvel[:] = 0
        data.qacc[:] = 0

        # data.qpos[pelvis_tilt_id] -= 0.02
        # data.qpos[ankle_angle_r_id] -= 0.01
        # data.qpos[ankle_angle_l_id] -= 0.01

        mj_forward(model, data)

        for step in range(sim_steps):
            buf_idx = step % buffer_size

            l = data.actuator_length[actuator_ids]
            v = data.actuator_velocity[actuator_ids]
            f = data.actuator_force[actuator_ids]

            length_buffer[buf_idx] = l
            velocity_buffer[buf_idx] = v
            force_buffer[buf_idx] = f

            if step >= delay_steps:
                delayed_idx = (step - delay_steps) % buffer_size
                l_delayed = length_buffer[delayed_idx]
                v_delayed = velocity_buffer[delayed_idx]
                f_delayed = force_buffer[delayed_idx]

                # 積分要素については、必要な時に以下のcontrollerを修正
                u = controller(Kp, l0, l_delayed, Kd, v0, v_delayed, Kf, f0, f_delayed, ff)

            else:
                u = ff.copy()


            u += np.random.normal(0.0, noise_std * u, size=u.shape)
            np.clip(u, 0.0, 1.0, out=u)

            # u += np.random.normal(0.0, 0.01, size=u.shape)
            # np.clip(u, 0.0, 1.0, out=u)


            # set ctrl vector using precomputed actuator ids
            data.ctrl[:] = 0.0
            data.ctrl[actuator_ids] = u

            mj_step(model, data)

            com = data.subtree_com[1]

            com_log[step] = com

            # --- 転倒判定 ---
            if float(com[2]) < 0.90 * com_init_height and fallen == False:
                fall_cost = float(sim_steps - step)
                fallen = True

            a = data.act[actuator_ids]

            total_Edot += Wang2012(u, a, l, v, f, muscle_mass, l_opt, l_t, slow_twitch_ratio)
        
        logs = {
            "total_Edot": total_Edot,
            "com_log": com_log,
            "fall_cost": fall_cost,
            "sim_steps": sim_steps,
        }

        all_costs.append(evaluator(logs))

    return tuple(np.mean(np.asarray(all_costs), axis=0))