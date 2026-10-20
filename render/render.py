import numpy as np
import os
import imageio.v2 as imageio
from render.plot_muscle_length import plot_muscle_length
import mujoco
from mujoco import MjModel, MjData, mj_step, mj_forward

def render_video(model_path, x, muscles, controller, sim_steps, model_name, delay_time=0.0, noise_std=0.0, camera=None, opt_name=None):

    model = MjModel.from_xml_path(model_path)
    actuator_ids = [model.actuator(m).id for m in muscles]
    data = MjData(model)

    com_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "com_marker")
    cop_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cop_marker")
    pelvis_tilt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_tilt")
    ankle_angle_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_angle_r")
    ankle_angle_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_angle_l")

    n_muscles = len(muscles)
    zero = np.zeros(n_muscles)

    Kp = x.get("Kp", zero)
    l0 = x.get("l0", zero)
    Kd = x.get("Kd", zero)
    v0 = x.get("v0", zero)
    Kf = x.get("Kf", zero)
    f0 = x.get("f0", zero)
    ff = x.get("ff", zero)


    # 初期化
    data.qpos[:] = model.qpos0.copy()
    data.qvel[:] = 0
    data.qacc[:] = 0

    mj_forward(model, data)

    dt = model.opt.timestep
    delay_steps = max(0, int(delay_time / dt))
    buffer_size = delay_steps + 1

    length_buffer = np.zeros((buffer_size, n_muscles))
    velocity_buffer = np.zeros((buffer_size, n_muscles))
    force_buffer = np.zeros((buffer_size, n_muscles))

    renderer = mujoco.Renderer(model, height=400, width=400)
    frames = []

    plot_names = list(muscles.keys())

    # 記録用リスト
    log_len = {name: [] for name in plot_names}
    log_target = {name: [] for name in plot_names}
    log_velocity= {name: [] for name in plot_names}
    log_u = {name: [] for name in plot_names}
    log_act = {name: [] for name in plot_names}


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

        # set ctrl vector using precomputed actuator ids
        data.ctrl[:] = 0.0
        data.ctrl[actuator_ids] = u

        # COM 計算
        com = data.subtree_com[1]

        # geom の位置を更新
        model.geom_pos[com_geom_id] = com

        # 接触している geom の確認
        contacts = []

        total_Fz = 0.0
        weighted_pos = np.zeros(3)

        for i in range(data.ncon):
            con = data.contact[i]
            g1, g2 = con.geom1, con.geom2
            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
            contacts.append((name1, name2))

            pos = con.pos.copy()

            # 床反力
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)

            # force = [fx, fy, fz, mx, my, mz]
            fz = force[0]

            # print(f"t={t}, contact: {name1} -- {name2}, Fz={fz:.4f}, pos={pos}")

            if fz > 0:   # 垂直方向に押している接触のみ採用
                weighted_pos += pos * abs(fz)
                total_Fz += abs(fz)

        if total_Fz > 0:
            cop = weighted_pos / total_Fz
            cop[2] = 0.0
            model.geom_pos[cop_geom_id] = cop
            # print(cop)
            model.geom_rgba[cop_geom_id] = np.array([0, 1, 0, 1])
        else:
            cop = np.array([np.nan, np.nan, np.nan])
            model.geom_rgba[cop_geom_id] = np.array([1, 0, 0, 1])

        # 1 step
        mj_step(model, data)

        # 色付け
        # for idx, ui in zip(actuator_ids, u):
        #     model.tendon_rgba[idx] = np.array([float(ui), 0.0, float(1.0-ui), 1.0])

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

        for tid, ui in zip(tendon_ids, u):
            model.tendon_rgba[tid] = np.array([float(ui), 0.0, float(1.0 - ui), 1.0])

        a = data.act[actuator_ids]

        # render
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())

        # ==== ログ保存（全ての筋） ====
        for i, name in enumerate(plot_names):
            log_len[name].append(l[i])
            log_target[name].append(l0[i])
            log_velocity[name].append(v[i])
            log_u[name].append(u[i])
            log_act[name].append(a[i])


    video_dir = os.path.join("results", model_name, "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{model_name}_{opt_name}_{camera}.mp4")

    with imageio.get_writer(video_path, fps=500, codec="libx264", pixelformat="yuv420p") as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"[render_video] wrote {video_path} ({len(frames)} frames)")

    # ===== グラフ描画 ====
    logs = [log_len, log_target, log_u, log_act, log_velocity]
    labels = ["Length", "Target Length", "Excitation(u)", "Activation(a)", "Velocity"]
    plot_muscle_length(model_name, sim_steps, plot_names, logs, labels, opt_name)

