import numpy as np

def Wang2012(u, a, l_MTU, v_MTU, F_MTU, mass, l_opt, l_t, slow_twitch_ratio):
    """
    Simplified Wang2012 metabolic energy model
    for MyoLeg26 / MuJoCo muscle actuators.

    Parameters
    ----------
    u : ndarray
        Muscle excitation (= data.ctrl)

    a : ndarray
        Muscle activation (= data.act)

    l_MTU : ndarray
        MuJoCo actuator / MTU length
        (= data.actuator_length) [m]

    v_MTU : ndarray
        MuJoCo actuator / MTU velocity
        (= data.actuator_velocity) [m/s]

        shortening : v_MTU < 0
        lengthening: v_MTU > 0

    F_MTU : ndarray
        MuJoCo actuator force
        (= data.actuator_force) [N]

    mass : ndarray
        Muscle mass [kg]

    l_opt : ndarray
        MuJoCo optimal muscle length L0 [m]

    l_t : ndarray
        MuJoCo constant tendon-length equivalent LT [m]

    slow_twitch_ratio : ndarray
        Slow twitch fiber ratio [0-1]

    Returns
    -------
    total_energy : float
        Total metabolic power of all muscles [W]

    muscle_energy : ndarray
        Metabolic power of each muscle [W]
    """

    # ==========================================================
    # Convert to NumPy arrays
    # ==========================================================

    u = np.asarray(u, dtype=float)              # MuJoCo 励起度u
    a = np.asarray(a, dtype=float)              # MuJoCo 活性度a

    l_MTU = np.asarray(l_MTU, dtype=float)      # MuJoCo 筋長（筋長 + 腱長）
    v_MTU = np.asarray(v_MTU, dtype=float)      # MuJoCo 筋収縮速度
    F_MTU = np.asarray(F_MTU, dtype=float)      # MuJoCo 筋長力

    mass = np.asarray(mass, dtype=float)        # 筋の質量
    l_opt = np.asarray(l_opt, dtype=float)      # 筋の最適化長
    l_t = np.asarray(l_t, dtype=float)          # 腱長
    slow_twitch_ratio = np.asarray(slow_twitch_ratio, dtype=float)  # 遅筋割合

    # ==========================================================
    # CE length
    #
    # MuJoCo muscle model:
    #
    #   l_MTU = LT + l_CE
    #
    # where LT is treated as constant.
    #
    # Therefore:
    #
    #   l_CE = l_MTU - LT
    #
    # and normalized CE length is:
    #
    #   l_ce_norm = l_CE / L0
    #             = (l_MTU - LT) / L0
    # ==========================================================

    l_CE = l_MTU - l_t

    l_ce_norm = l_CE / l_opt

    # ==========================================================
    # CE velocity
    #
    # Since LT is constant:
    #
    #   d(LT)/dt = 0
    #
    # therefore:
    #
    #   v_CE = v_MTU
    #
    # shortening  -> v_CE < 0
    # lengthening -> v_CE > 0
    # ==========================================================

    v_CE = v_MTU

    # ==========================================================
    # CE force
    #
    # MuJoCo actuator_force can have a sign depending on the
    # actuator/transmission convention.
    #
    # Metabolic calculations require tensile force magnitude,
    # therefore use absolute value.
    # ==========================================================

    F_CE = np.abs(F_MTU)

    # ==========================================================
    # Slow twitch fiber ratio
    # ==========================================================

    s = slow_twitch_ratio

    # ==========================================================
    # Activation heat rate [W/kg]
    # ==========================================================

    fa = (40.0 * s * np.sin(np.pi / 2.0 * u) + 133.0 * (1.0 - s) * (1.0 - np.cos(np.pi / 2.0 * u)))

    # ==========================================================
    # Maintenance heat rate [W/kg]
    # ==========================================================

    fm = (74.0 * s * np.sin(np.pi / 2.0 * a) + 111.0 * (1.0 - s) * (1.0 - np.cos(np.pi / 2.0 * a)))

    # ==========================================================
    # Length dependency
    # ==========================================================

    g = np.where(
        l_ce_norm < 0.5,
        0.5,
        np.where(
            l_ce_norm < 1.0,
            l_ce_norm,
            np.where(
                l_ce_norm < 1.5,
                -2.0 * l_ce_norm + 3.0,
                0.0,
            ),
        ),
    )

    # ==========================================================
    # Activation heat [W]
    # ==========================================================

    effort_a = mass * fa

    # ==========================================================
    # Maintenance heat [W]
    # ==========================================================

    effort_m = mass * g * fm

    # ==========================================================
    # Shortening velocity
    #
    # v_CE < 0 means shortening.
    #
    # Convert shortening velocity to positive magnitude:
    #
    #   v_short = -v_CE  if v_CE < 0
    #             0      otherwise
    # ==========================================================

    v_short = np.maximum(-v_CE, 0.0)

    # ==========================================================
    # Shortening heat rate [W]
    #
    # F [N] * v [m/s] = W
    # ==========================================================

    effort_s = 0.25 * F_CE * v_short

    # ==========================================================
    # Positive mechanical work rate [W]
    #
    # Muscle produces positive mechanical work during
    # shortening.
    # ==========================================================

    effort_w = F_CE * v_short

    # ==========================================================
    # Total metabolic rate of each muscle [W]
    # ==========================================================

    muscle_energy = (effort_a + effort_m + effort_s + effort_w)

    # ==========================================================
    # Total metabolic rate [W]
    # ==========================================================

    total_energy = np.sum(muscle_energy)

    return total_energy