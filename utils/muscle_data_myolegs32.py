from dataclasses import dataclass
import numpy as np


# ==========================================================
# Muscle parameter class
# ==========================================================

from dataclasses import dataclass


# ==========================================================
# Muscle parameter class for MyoLeg26
# ==========================================================

@dataclass
class MuscleParameters:

    # maximum isometric force [N]
    #
    # MyoLeg26:
    #   actuator_gainprm[:, 2]
    f_max: float

    # MuJoCo optimal muscle length L0 [m]
    #
    # calculated from:
    #
    #   L0 = (length_max - length_min)
    #        / (r_max - r_min)
    #
    # where:
    #   r_min, r_max = actuator_gainprm[:, 0:2]
    #   length_min, length_max = actuator_lengthrange
    #
    # NOTE:
    # This is the MuJoCo muscle-model parameter L0.
    # It is not necessarily identical to the physiological
    # optimal_fiber_length used in OpenSim.
    l_opt: float

    # maximum contraction velocity [L0 / s]
    #
    # MyoLeg26:
    #   actuator_gainprm[:, 6]
    v_max: float

    # slow twitch fiber ratio [0-1]
    #
    # MyoLeg26 itself does not contain this parameter.
    # Values are supplied from external physiological /
    # OpenSim metabolic-model data.
    # cf. https://github.com/opensim-org/opensim-models/blob/master/Tutorials/Design_to_Reduce_Metabolic_Cost/Scripts/metabolicsSlowTwitchRatios_Gait2392.txt
    slow_twitch_ratio: float

    # muscle mass [kg]
    #
    # Used in the metabolic energy model.
    #
    # NOTE:
    # MyoLeg26 itself does not explicitly contain
    # physiological muscle mass.
    #
    # If estimated from:
    #
    #   muscle_mass = (f_max / specific_tension) * density * l_opt
    #   specific_tension = 0.25e6 [N/m^2]
    #   density = 1059.7 [kg/m^3]
    #
    # care is required because MuJoCo L0 is not necessarily
    # identical to physiological optimal fiber length.
    # cf. https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1Umberger2010MuscleMetabolicsProbe__MetabolicMuscleParameter.html
    muscle_mass: float

    # MuJoCo constant tendon-length equivalent LT [m]
    #
    # calculated from:
    #
    #   LT = length_min - r_min * L0
    #
    # MuJoCo normalized contractile-element length:
    #
    #   l_ce_norm = (l_MTU - LT) / L0
    #
    # NOTE:
    # This is NOT necessarily identical to the physiological
    # tendon_slack_length used in OpenSim.
    l_t: float


# ==========================================================
# MyoLeg26 muscle parameters
#
# IMPORTANT:
# right and left muscles are intentionally stored separately
# because future asymmetric parameter tuning may be required.
#
# NOTE:
# - f_max, l_opt, v_max, and l_t are based on the
#   MyoLeg26 MuJoCo muscle model.
# - slow_twitch_ratio is supplied externally.
# - muscle_mass is a parameter for the metabolic model
#   and should not automatically be interpreted as a
#   directly measured physiological muscle mass.
# ==========================================================


muscle_data = {

    # ======================================================
    # abd
    # ======================================================

    "abd_r": MuscleParameters(
        f_max=4323.26,
        l_opt=0.109295,
        v_max=10.0,
        slow_twitch_ratio=0.500,
        muscle_mass=2.002874,
        l_t=0.035152,
    ),

    "abd_l": MuscleParameters(
        f_max=4323.26,
        l_opt=0.109295,
        v_max=10.0,
        slow_twitch_ratio=0.500,
        muscle_mass=2.002874,
        l_t=0.035152,
    ),


    # ======================================================
    # add
    # ======================================================

    "add_r": MuscleParameters(
        f_max=3871.82,
        l_opt=0.129623,
        v_max=10.0,
        slow_twitch_ratio=0.500,
        muscle_mass=2.127355,
        l_t=0.029445,
    ),

    "add_l": MuscleParameters(
        f_max=3871.82,
        l_opt=0.129623,
        v_max=10.0,
        slow_twitch_ratio=0.500,
        muscle_mass=2.127355,
        l_t=0.029445,
    ),


    # ======================================================
    # hamstrings
    # ======================================================

    "hamstrings_r": MuscleParameters(
        f_max=3948.49,
        l_opt=0.128139,
        v_max=10.0,
        slow_twitch_ratio=0.499,
        muscle_mass=2.144647,
        l_t=0.308430,
    ),

    "hamstrings_l": MuscleParameters(
        f_max=3948.49,
        l_opt=0.128139,
        v_max=10.0,
        slow_twitch_ratio=0.499,
        muscle_mass=2.144647,
        l_t=0.308430,
    ),


    # ======================================================
    # bifemsh
    # ======================================================

    "bifemsh_r": MuscleParameters(
        f_max=509.42,
        l_opt=0.101662,
        v_max=10.0,
        slow_twitch_ratio=0.529,
        muscle_mass=0.219524,
        l_t=0.122387,
    ),

    "bifemsh_l": MuscleParameters(
        f_max=509.42,
        l_opt=0.101662,
        v_max=10.0,
        slow_twitch_ratio=0.529,
        muscle_mass=0.219524,
        l_t=0.122387,
    ),


    # ======================================================
    # edl
    # ======================================================

    "edl_r": MuscleParameters(
        f_max=553.24,
        l_opt=0.069301,
        v_max=10.0,
        slow_twitch_ratio=0.500,
        muscle_mass=0.162515,
        l_t=0.368875,
    ),

    "edl_l": MuscleParameters(
        f_max=553.24,
        l_opt=0.069301,
        v_max=10.0,
        slow_twitch_ratio=0.500,
        muscle_mass=0.162515,
        l_t=0.368875,
    ),


    # ======================================================
    # fdl
    # ======================================================

    "fdl_r": MuscleParameters(
        f_max=332.13,
        l_opt=0.044600,
        v_max=10.0,
        slow_twitch_ratio=0.500,
        muscle_mass=0.062789,
        l_t=0.378773,
    ),

    "fdl_l": MuscleParameters(
        f_max=332.13,
        l_opt=0.044600,
        v_max=10.0,
        slow_twitch_ratio=0.500,
        muscle_mass=0.062789,
        l_t=0.378773,
    ),


    # ======================================================
    # glutmax
    # ======================================================

    "glutmax_r": MuscleParameters(
        f_max=3276.04,
        l_opt=0.166891,
        v_max=10.0,
        slow_twitch_ratio=0.550,
        muscle_mass=2.317531,
        l_t=0.040509,
    ),

    "glutmax_l": MuscleParameters(
        f_max=3276.04,
        l_opt=0.166891,
        v_max=10.0,
        slow_twitch_ratio=0.550,
        muscle_mass=2.317531,
        l_t=0.040509,
    ),


    # ======================================================
    # iliopsoas
    # ======================================================

    "iliopsoas_r": MuscleParameters(
        f_max=2585.58,
        l_opt=0.139974,
        v_max=10.0,
        slow_twitch_ratio=0.500,
        muscle_mass=1.534083,
        l_t=0.116366,
    ),

    "iliopsoas_l": MuscleParameters(
        f_max=2585.58,
        l_opt=0.139974,
        v_max=10.0,
        slow_twitch_ratio=0.500,
        muscle_mass=1.534083,
        l_t=0.116366,
    ),


    # ======================================================
    # rectfem
    # ======================================================

    "rectfem_r": MuscleParameters(
        f_max=1800.50,
        l_opt=0.149262,
        v_max=10.0,
        slow_twitch_ratio=0.387,
        muscle_mass=1.139163,
        l_t=0.304589,
    ),

    "rectfem_l": MuscleParameters(
        f_max=1800.50,
        l_opt=0.149262,
        v_max=10.0,
        slow_twitch_ratio=0.387,
        muscle_mass=1.139163,
        l_t=0.304589,
    ),


    # ======================================================
    # vasti
    # ======================================================

    "vasti_r": MuscleParameters(
        f_max=9500.94,
        l_opt=0.100486,
        v_max=10.0,
        slow_twitch_ratio=0.484,
        muscle_mass=4.046815,
        l_t=0.105259,
    ),

    "vasti_l": MuscleParameters(
        f_max=9500.94,
        l_opt=0.100486,
        v_max=10.0,
        slow_twitch_ratio=0.484,
        muscle_mass=4.046815,
        l_t=0.105259,
    ),


    # ======================================================
    # gastroc
    # ======================================================

    "gastroc_r": MuscleParameters(
        f_max=3820.44,
        l_opt=0.128843,
        v_max=10.0,
        slow_twitch_ratio=0.546,
        muscle_mass=2.086492,
        l_t=0.347701,
    ),

    "gastroc_l": MuscleParameters(
        f_max=3820.44,
        l_opt=0.128843,
        v_max=10.0,
        slow_twitch_ratio=0.546,
        muscle_mass=2.086492,
        l_t=0.347701,
    ),


    # ======================================================
    # soleus
    # ======================================================

    "soleus_r": MuscleParameters(
        f_max=6090.00,
        l_opt=0.092898,
        v_max=10.0,
        slow_twitch_ratio=0.759,
        muscle_mass=2.398095,
        l_t=0.216649,
    ),

    "soleus_l": MuscleParameters(
        f_max=6090.00,
        l_opt=0.092898,
        v_max=10.0,
        slow_twitch_ratio=0.759,
        muscle_mass=2.398095,
        l_t=0.216649,
    ),


    # ======================================================
    # tibant
    # ======================================================

    "tibant_r": MuscleParameters(
        f_max=2042.40,
        l_opt=0.098127,
        v_max=10.0,
        slow_twitch_ratio=0.721,
        muscle_mass=0.849518,
        l_t=0.225262,
    ),

    "tibant_l": MuscleParameters(
        f_max=2042.40,
        l_opt=0.098127,
        v_max=10.0,
        slow_twitch_ratio=0.721,
        muscle_mass=0.849518,
        l_t=0.225262,
    ),

    # ======================================================
    # ercspn
    # ======================================================
    
    "ercspn_r": MuscleParameters(
        f_max=2458.55,
        l_opt=0.172827,
        v_max=10.0,
        slow_twitch_ratio=0.6,
        muscle_mass=1.801079,
        l_t=-0.008584,
    ),

    "ercspn_l": MuscleParameters(
        f_max=2458.55,
        l_opt=0.172827,
        v_max=10.0,
        slow_twitch_ratio=0.6,
        muscle_mass=1.801079,
        l_t=-0.008584,
    ),


    # ======================================================
    # intobl
    # ======================================================
    
    "intobl_r": MuscleParameters(
        f_max=1304.15,
        l_opt=0.187567,
        v_max=10.0,
        slow_twitch_ratio=0.56,
        muscle_mass=1.036874,
        l_t=0.006044,
    ),

    "intobl_l": MuscleParameters(
        f_max=1304.15,
        l_opt=0.187567,
        v_max=10.0,
        slow_twitch_ratio=0.56,
        muscle_mass=1.036874,
        l_t=0.006044,
    ),

    # ======================================================
    # extobl
    # ======================================================
    
    "extobl_r": MuscleParameters(
        f_max=853.68,
        l_opt=0.279930,
        v_max=10.0,
        slow_twitch_ratio=0.58,
        muscle_mass=1.012946,
        l_t=0.015648,
    ),

    "extobl_l": MuscleParameters(
        f_max=853.68,
        l_opt=0.279930,
        v_max=10.0,
        slow_twitch_ratio=0.58,
        muscle_mass=1.012946,
        l_t=0.015648,
    ),

}

# ==========================================================
# utility function
#
# convert dictionary-based parameters into NumPy arrays
# for vectorized computation
# ==========================================================

def get_parameter_array(muscle_names, parameter_name):

    return np.array([
        getattr(muscle_data[name], parameter_name)
        for name in muscle_names
    ])

def get_pairs():

    pairs = []

    for name in muscle_data.keys():

        if name.endswith("_r"):

            left_name = name[:-2] + "_l"

            if left_name in muscle_data:
                pairs.append((name, left_name))

    return pairs

def get_right_muscles():

    return [
        m
        for m in muscle_data.keys()
        if m.endswith("_r")
    ]
    
def get_left_muscles():

    return [
        m
        for m in muscle_data.keys()
        if m.endswith("_l")
    ]