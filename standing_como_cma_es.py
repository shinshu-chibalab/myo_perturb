import numpy as np
from optim.como_cma_es import COMO_CMA_ES
from render.render import render_video
from controller.Standing_Controller import standing_controller
from evaluator.f1Energy_f2Com_evaluator import standing_evaluator
from x0_myolegs26 import *

from cos_sim import analyze_vectors


if __name__ == "__main__":

    model_path = "myo_sim\models\leg\myolegs26.xml"
    model_name = "myolegs26_standing_v1(como_cma_es)"
    muscles = myolegs26_v0

    sim_steps = 1000
    sigma0 = 0.01
    popsize = 896
    maxiter = 3000
    delay_time = 0.0
    noise_std = 0.0
    n_jobs = 56
    symmetry = True
    n_kernels = 8
    reference_point=(3e3, 3e3)

    controller = standing_controller
    evaluator = standing_evaluator

    optimizer = COMO_CMA_ES(
        model_path=model_path,
        muscles=muscles,
        controller=controller,
        evaluator=evaluator,
        sim_steps=sim_steps,
        model_name=model_name,
    )

    result = optimizer.optimize(
        sigma0=sigma0,
        maxiter=maxiter,
        popsize=popsize,
        delay_time=delay_time,
        noise_std=noise_std,
        n_jobs=n_jobs,
        symmetry=symmetry,
        n_kernels=n_kernels,
        reference_point=reference_point,
    )

    best_params = result["params"]
    best_fitness = result["fitness"]

    print(
        f"model_path: {model_path}\n"
        f"model_name: {model_name}\n"
        f"sim_steps:  {sim_steps}\n"
        f"popsize:    {popsize}\n"
        f"maxiter:    {maxiter}\n"
        f"delay_time: {delay_time}\n"
        f"noise_std:  {noise_std}\n"
        f"n_jobs:     {n_jobs}\n"
        f"symmetry:   {symmetry}\n"
    )

    cameras = ["front_view", "side_view"]

    ff_list = []
    Kp_list = []
    Kd_list = []
    muscle_len_list = []

    for i, (params, fitness) in enumerate(zip(best_params, best_fitness), start=1):

        fit_text = "_".join(
            f"f{j+1}_{value}"
            for j, value in enumerate(fitness)
        )

        opt_name = f"{fit_text}_res{i}"

        for camera in cameras:
            render_video(
                model_path=model_path,
                x=params,
                muscles=muscles,
                controller=controller,
                sim_steps=sim_steps,
                model_name=model_name,
                delay_time=delay_time,
                noise_std=noise_std,
                camera=camera,
                opt_name=opt_name,
            )

        print(f"\n[Solution {i}]")

        for j, value in enumerate(fitness, start=1):
            print(f"f{j}: {value}")

        print("l0:", params.get("l0"))
        print("Kp:", params.get("Kp"))
        print("Kd:", params.get("Kd"))
        print("ff:", params.get("ff"))

        ff_list.append(params.get("ff"))
        Kp_list.append(params.get("Kp"))
        Kd_list.append(params.get("Kd"))
        muscle_len_list.append(params.get("l0"))

    analyze_vectors(ff_list, "ff", model_name)
    analyze_vectors(Kp_list, "Kp", model_name)
    analyze_vectors(Kd_list, "Kd", model_name)
    analyze_vectors(muscle_len_list, "muscle_len", model_name)