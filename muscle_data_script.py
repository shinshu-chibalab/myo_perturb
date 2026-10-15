from dataclasses import dataclass
import mujoco
import myo_sim
import numpy as np


# ==========================================================
# Muscle parameter class
# ==========================================================

@dataclass
class MuscleParameters:

    # maximum isometric force [N]
    f_max: float

    # MuJoCo optimal muscle length L0 [m]
    l_opt: float

    # maximum contraction velocity [l_opt / s]
    v_max: float

    # slow twitch fiber ratio [0-1]
    slow_twitch_ratio: float

    # muscle mass [kg]
    #
    # muscle_mass =
    #     (f_max / specific_tension) * density * l_opt
    #
    # specific_tension = 0.25e6 [N/m^2]
    # density = 1059.7 [kg/m^3]
    muscle_mass: float

    # MuJoCo constant tendon-length equivalent LT [m]
    #
    # NOTE:
    # This is not strictly identical to OpenSim tendon_slack_length.
    l_t: float


# ==========================================================
# Constants for muscle mass calculation
# ==========================================================

SPECIFIC_TENSION = 0.25e6   # [N/m^2]
MUSCLE_DENSITY = 1059.7     # [kg/m^3]


# ==========================================================
# Slow twitch fiber ratio
#
# MyoLeg26 itself does not contain these parameters.
# Values are supplied separately.
#
# Muscles for which a corresponding value is not available
# are temporarily assigned 0.50.
# ==========================================================

SLOW_TWITCH_RATIO = {

    "abd": 0.50,
    "add": 0.50,

    "hamstrings": 0.499,
    "bifemsh": 0.529,

    "edl": 0.50,
    "fdl": 0.50,

    "glutmax": 0.550,
    "iliopsoas": 0.500,

    "rectfem": 0.387,
    "vasti": 0.484,

    "gastroc": 0.546,
    "soleus": 0.759,
    "tibant": 0.721,
}


# ==========================================================
# Utility functions
# ==========================================================

def get_base_muscle_name(name: str) -> str:
    """
    Remove the left/right suffix.

    Example:
        hamstrings_r -> hamstrings
        soleus_l     -> soleus
    """

    if name.endswith("_r") or name.endswith("_l"):
        return name[:-2]

    return name


def get_mujoco_muscle_parameters(model, actuator_id):
    """
    Extract muscle parameters from a MuJoCo muscle actuator.

    gainprm:
        [0] range_min
        [1] range_max
        [2] Fmax
        [3] scale
        [4] lmin (FLV parameter)
        [5] lmax (FLV parameter)
        [6] vmax
        [7] fpmax
        [8] fvmax

    Returns:
        f_max
        l_opt (= L0)
        v_max
        l_t (= LT)
    """

    # ------------------------------------------------------
    # MuJoCo muscle parameters
    # ------------------------------------------------------

    gainprm = model.actuator_gainprm[actuator_id]

    r_min = gainprm[0]
    r_max = gainprm[1]

    f_max = gainprm[2]
    v_max = gainprm[6]


    # ------------------------------------------------------
    # Actuator / MTU length range [m]
    # ------------------------------------------------------

    length_min = model.actuator_lengthrange[actuator_id, 0]
    length_max = model.actuator_lengthrange[actuator_id, 1]


    # ------------------------------------------------------
    # Optimal muscle length L0
    #
    # length = LT + L0 * normalized_length
    #
    # Therefore:
    #
    # L0 =
    #   (length_max - length_min)
    #   / (r_max - r_min)
    # ------------------------------------------------------

    l_opt = (
        (length_max - length_min)
        / (r_max - r_min)
    )


    # ------------------------------------------------------
    # Constant tendon-length equivalent LT
    #
    # length_min = LT + r_min * L0
    #
    # therefore:
    #
    # LT = length_min - r_min * L0
    # ------------------------------------------------------

    l_t = length_min - r_min * l_opt


    return f_max, l_opt, v_max, l_t


def calculate_muscle_mass(
    f_max,
    l_opt,
    specific_tension=SPECIFIC_TENSION,
    density=MUSCLE_DENSITY,
):
    """
    Calculate muscle mass [kg].

    PCSA = Fmax / specific_tension

    muscle volume ≈ PCSA * l_opt

    muscle mass = volume * density
    """

    pcsa = f_max / specific_tension

    muscle_volume = pcsa * l_opt

    muscle_mass = muscle_volume * density

    return muscle_mass


# ==========================================================
# Create muscle_data
# ==========================================================

def create_muscle_data(model):

    muscle_data = {}

    for actuator_id in range(model.nactuator):

        # --------------------------------------------------
        # Muscle name
        # --------------------------------------------------

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )

        if name is None:
            continue


        # --------------------------------------------------
        # Parameters obtained from MyoLeg26
        # --------------------------------------------------

        f_max, l_opt, v_max, l_t = \
            get_mujoco_muscle_parameters(
                model,
                actuator_id,
            )


        # --------------------------------------------------
        # Slow twitch ratio
        # --------------------------------------------------

        base_name = get_base_muscle_name(name)

        if base_name not in SLOW_TWITCH_RATIO:
            raise KeyError(
                f"slow twitch ratio is not defined for: {name}"
            )

        slow_twitch_ratio = \
            SLOW_TWITCH_RATIO[base_name]


        # --------------------------------------------------
        # Muscle mass
        # --------------------------------------------------

        muscle_mass = calculate_muscle_mass(
            f_max=f_max,
            l_opt=l_opt,
        )


        # --------------------------------------------------
        # Store
        # --------------------------------------------------

        muscle_data[name] = MuscleParameters(

            f_max=float(f_max),

            l_opt=float(l_opt),

            v_max=float(v_max),

            slow_twitch_ratio=float(
                slow_twitch_ratio
            ),

            muscle_mass=float(
                muscle_mass
            ),

            l_t=float(l_t),
        )

    return muscle_data


# ==========================================================
# Utility functions
# ==========================================================

def get_parameter_array(
    muscle_data,
    muscle_names,
    parameter_name,
):

    return np.array([
        getattr(
            muscle_data[name],
            parameter_name,
        )
        for name in muscle_names
    ])


def get_pairs(muscle_data):

    pairs = []

    for name in muscle_data:

        if name.endswith("_r"):

            left_name = name[:-2] + "_l"

            if left_name in muscle_data:

                pairs.append(
                    (name, left_name)
                )

    return pairs


def get_right_muscles(muscle_data):

    return [
        name
        for name in muscle_data
        if name.endswith("_r")
    ]


def get_left_muscles(muscle_data):

    return [
        name
        for name in muscle_data
        if name.endswith("_l")
    ]


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    model, data = myo_sim.load(
        "myolegs26"
    )

    muscle_data = create_muscle_data(
        model
    )


    # ------------------------------------------------------
    # Show parameters
    # ------------------------------------------------------

    for name, param in muscle_data.items():

        print(
            f"{name:15s} "
            f"Fmax={param.f_max:8.2f} N  "
            f"L0={param.l_opt:.6f} m  "
            f"LT={param.l_t:.6f} m  "
            f"Vmax={param.v_max:.2f}  "
            f"slow={param.slow_twitch_ratio:.3f}  "
            f"mass={param.muscle_mass:.6f} kg"
        )