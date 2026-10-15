import numpy as np
import random
import os
import copy
from deap import base, creator, tools, algorithms
from functools import partial
from multiprocessing import Pool

from mujoco import MjModel, MjData, mj_forward
import mujoco

from utils.param_utils import create_x0_bounds, expand_params
from optim.base import OptimizerBase
from simulation.worker import run_simulation_worker, _init_worker
from render.plot_cost import plot_cost_history
from render.plot_pareto import plot_pareto_front, plot_pareto_front_hist
from render.render import render_video

class NSGA2(OptimizerBase):
    def __init__(self, model_path, muscles, controller, evaluator, sim_steps, model_name):
        self.model_path = model_path
        self.model = MjModel.from_xml_path(model_path)
        self.muscles = muscles
        self.actuator_ids = [self.model.actuator(m).id for m in muscles]
        self.controller = controller
        self.evaluator = evaluator

        data = MjData(self.model)
        data.qpos[:] = self.model.qpos0.copy()
        data.qvel[:] = 0
        data.qacc[:] = 0
        mj_forward(self.model, data)

        self.com_init_height = float(np.array(data.subtree_com[1])[2])
        self.sim_steps = sim_steps
        self.model_name = model_name

        self.cost_history_min1 = []
        self.cost_history_min2 = []
        self.cost_history_mean1 = []
        self.cost_history_mean2 = []

    def _evaluate(self, individual):
        f1, f2 = run_simulation_worker(
            individual,
            controller=self.controller,
            evaluator=self.evaluator,
            delay_time=self.delay_time,
            noise_std=self.noise_std,
        )

        return (f1, f2)  # ← DEAPは tuple
    
    def _perturb_x0(self, x0, ratio=0.01):
        x0 = np.asarray(x0)
        eps = np.random.uniform(-ratio, ratio, size=len(x0))
        return x0 * (1.0 + eps)
    
    @staticmethod
    def mut_percent(ind, bounds_lower, bounds_upper, indpb=0.1, local_scale=(0.95, 1.05), global_scale=(0.4, 1.6), global_prob=0.3):

        for i in range(len(ind)):
            if random.random() < indpb:

                if random.random() < global_prob:
                    scale = random.uniform(global_scale[0], global_scale[1])
                else:
                    scale = random.uniform(local_scale[0], local_scale[1])

                ind[i] *= scale

                ind[i] = float(np.clip(
                    ind[i],
                    bounds_lower[i],
                    bounds_upper[i]
                ))

        return (ind,)

    @staticmethod
    def get_search_schedule(gen, maxiter):
        progress = gen/(maxiter-1)

        cxpb = 0.7 - 0.2*progress

        mutpb = 0.6 - 0.1*progress

        indpb = 0.15 - 0.12*progress

        global_prob = 0.30 - 0.28*progress

        local_width = 0.10 - 0.085*progress

        local_scale = (
            1-local_width,
            1+local_width
        )

        global_width = 0.4 - 0.3*progress

        global_scale = (
            1-global_width,
            1+global_width
        )

        return cxpb, mutpb, indpb, local_scale, global_scale, global_prob

    
    def save_pareto_front_npy(self, pareto_front):
        save_path = os.path.join("results", self.model_name, "pareto_front.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        data = []
        for ind in pareto_front:
            f1, f2 = ind.fitness.values
            params = np.array(ind)
            data.append((f1, f2, params))

        np.save(save_path, np.array(data, dtype=object), allow_pickle=True)

    @staticmethod
    def select_diverse_pareto(pareto_front, n_select=10):

        if len(pareto_front) <= n_select:
            return pareto_front

        f = np.array([ind.fitness.values for ind in pareto_front])

        f_min = f.min(axis=0)
        f_max = f.max(axis=0)

        X = (f - f_min) / (f_max - f_min + 1e-12)

        idx_min = np.argmin(X[:,0])
        idx_max = np.argmax(X[:,0])

        selected_idx = [idx_min, idx_max]
        remaining = list(set(range(len(pareto_front))) - set(selected_idx))

        while len(selected_idx) < n_select:

            best_idx = None
            best_dist = -1

            for i in remaining:

                d = min(np.linalg.norm(X[i] - X[j]) for j in selected_idx)

                if d > best_dist:
                    best_dist = d
                    best_idx = i

            selected_idx.append(best_idx)
            remaining.remove(best_idx)

        return [pareto_front[i] for i in selected_idx]

    def optimize(self, sigma0=1, maxiter=50, popsize=100, delay_time=0.0, noise_std=0.0, n_jobs=1, symmetry=True):
        
        x0, bounds_lower, bounds_upper = create_x0_bounds(self.muscles, symmetry=symmetry)

        self.delay_time = delay_time
        self.noise_std = noise_std

        # ===== DEAP setup =====
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register("evaluate", self._evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("mutate", self.mut_percent, bounds_lower=bounds_lower, bounds_upper=bounds_upper, indpb=0.1)

        # ===== multiprocessing =====
        pool = Pool(
            processes=n_jobs,
            initializer=_init_worker,
            initargs=(
                self.model_path,
                self.muscles,
                self.sim_steps,
                self.model.qpos0,
                self.com_init_height,
                symmetry,
            )
        )
        
        toolbox.register("map", pool.map, chunksize=4)

        # ===== 初期集団 =====

        pop_x0 = [
            creator.Individual(
                np.clip(x0, bounds_lower, bounds_upper).tolist()
            )
        ]

        # 80% は広くランダム
        pop_random = [
            creator.Individual(
                np.random.uniform(bounds_lower, bounds_upper).tolist()
            )
            for _ in range(int(0.80 * popsize))
        ]

        # 20% は x0 近傍
        pop_local = [
            creator.Individual(
                np.clip(
                    self._perturb_x0(x0, ratio=0.05),
                    bounds_lower,
                    bounds_upper
                ).tolist()
            )
            for _ in range(popsize - len(pop_random) - len(pop_x0))
        ]

        pop = pop_x0 + pop_random + pop_local

        fitnesses = toolbox.map(toolbox.evaluate, pop)

        for ind, fit in zip(pop, fitnesses):
            f1, f2 = fit

            ind.fitness.values = (f1, f2)

        pop = toolbox.select(pop, len(pop))

        self.pareto_front = pop

        # 各世代ごとのenergy_costが低いパラメータを入れる配列
        params1 = []
        front1_hist = []

        # ===== evolution =====
        try:

            for gen in range(maxiter):

                cxpb, mutpb, indpb, local_scale, global_scale, global_prob = self.get_search_schedule(gen, maxiter)

                toolbox.unregister("mutate")
                toolbox.register(
                    "mutate",
                    self.mut_percent,
                    bounds_lower=bounds_lower,
                    bounds_upper=bounds_upper,
                    indpb=indpb,
                    local_scale=local_scale,
                    global_scale=global_scale,
                    global_prob=global_prob,
                )

                offspring = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)

                for ind in offspring:
                    for i in range(len(ind)):
                        ind[i] = float(np.clip(ind[i], bounds_lower[i], bounds_upper[i]))

                fitnesses = toolbox.map(toolbox.evaluate, offspring)
                
                for ind, fit in zip(offspring, fitnesses):
                    f1, f2 = fit

                    ind.fitness.values = (f1, f2)

                pop = toolbox.select(pop + offspring, popsize)

                self.fronts = tools.sortNondominated(
                    pop, k=len(pop), first_front_only=False
                )

                self.pareto_front = self.fronts[0]

                f1 = [ind.fitness.values[0] for ind in self.pareto_front]
                f2 = [ind.fitness.values[1] for ind in self.pareto_front]

                print(
                    f"Gen {gen:03d} | Pareto size={len(self.pareto_front)} | "
                    f"f1[min]={min(f1)}, f2[min]={min(f2)}"
                )

                self.cost_history_min1.append(np.min(f1))
                self.cost_history_min2.append(np.min(f2))
                self.cost_history_mean1.append(np.mean(f1))
                self.cost_history_mean2.append(np.mean(f2))

                if gen % 30 == 0:
                    self.save_pareto_front_npy(self.pareto_front)
                    best1 = min(self.pareto_front, key=lambda ind: ind.fitness.values[0])
                    self.best_params1 = self.best_params1 = expand_params(np.array(best1), self.muscles, symmetry=symmetry)
                    params1.append(self.best_params1)
                    front_fitness = np.array([ind.fitness.values for ind in self.pareto_front])
                    front1_hist.append((gen, front_fitness))
                    print(
                        f"schedule | cxpb={cxpb:.3f}, mutpb={mutpb:.3f}, "
                        f"indpb={indpb:.3f}, local_scale={local_scale}, "
                        f"global_scale={global_scale}, global_prob={global_prob:.3f}"
                    )

            
        except KeyboardInterrupt:

            print("\nOptimization interrupted by user")

        finally:

            pool.close()
            pool.join()

        # ===== best solution =====
        selected_pareto = self.select_diverse_pareto(
            self.pareto_front,
            n_select=10
        )

        selected_params = []

        for ind in selected_pareto:
            selected_params.append(
                expand_params(
                    np.array(ind),
                    self.muscles,
                    symmetry=symmetry,
                )
            )


        print(f"selected pareto solutions: {len(selected_pareto)}")

        for i, ind in enumerate(selected_pareto):
            res_f1, res_f2 = ind.fitness.values
            print(f"selected {i}: f1={res_f1}, f2={res_f2}")

        self.save_pareto_front_npy(self.pareto_front)

        for i, param in enumerate(params1):
            render_video(self.model_path, param, self.muscles, self.controller, self.sim_steps, self.model_name, delay_time, noise_std, camera="diagonal", opt_name=f"gen{(i)*10}")
        plot_pareto_front_hist(self.model_name, front1_hist[3:])
        print(f"plot pareto point 1: {len(self.pareto_front)}")
        plot_cost_history(self.model_name, self.cost_history_mean1, "mean_energy")
        plot_cost_history(self.model_name, self.cost_history_min1, "min_energy")
        plot_cost_history(self.model_name, self.cost_history_mean2, "mean_com")
        plot_cost_history(self.model_name, self.cost_history_min2, "min_com")
        plot_pareto_front(self.model_name, [self.fronts[0]])
        plot_pareto_front(self.model_name, self.fronts[:5])

        return {
            "params": selected_params,
            "fitness": [
                ind.fitness.values
                for ind in selected_pareto
            ]
        }
            
