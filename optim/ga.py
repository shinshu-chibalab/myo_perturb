import numpy as np
import random
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

class GA(OptimizerBase):
    def __init__(self, model_path, muscles, controller, evaluator, sim_steps, model_name):
        self.model_path = model_path
        self.model = MjModel.from_xml_path(model_path)  # メインプロセスでも一度ロードして情報取得
        self.muscles = muscles
        self.actuator_ids = [self.model.actuator(m).id for m in muscles]
        self.length_range = self.model.actuator_lengthrange[self.actuator_ids]
        self.controller = controller
        self.evaluator = evaluator

        data = MjData(self.model)
        data.qpos[:] = self.model.qpos0.copy()
        data.qvel[:] = 0
        data.qacc[:] = 0
        mj_forward(self.model, data)
        self.com_init_height = float(np.array(data.subtree_com[1])[2])

        self.init_len = np.array([data.actuator_length[act_id] for act_id in self.actuator_ids])

        self.sim_steps = sim_steps

        self.model_name = model_name

        self.cost_history_min = []
        self.cost_history_mean = []

    def _evaluate(self, individual):
        fitness = run_simulation_worker(
            individual,
            controller=self.controller,
            evaluator=self.evaluator,
            delay_time=self.delay_time,
            noise_std=self.noise_std,
        )

        return (float(fitness[0]),) # ← DEAPは tuple
    
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

    def optimize(self, sigma0=1, maxiter=50, popsize=100, delay_time=0.0, noise_std=0.0, n_jobs=1, symmetry=True):

        x0, bounds_lower, bounds_upper = create_x0_bounds(self.muscles, symmetry=symmetry)

        self.delay_time = delay_time
        self.noise_std = noise_std

        # ===== DEAP setup =====
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("evaluate", self._evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("select", tools.selTournament, tournsize=3)
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
            ind.fitness.values = fit

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
                    ind.fitness.values = fit

                pop = toolbox.select(pop + offspring, popsize)

                costs = [ind.fitness.values[0] for ind in pop]
                self.cost_history_min.append(np.min(costs))
                self.cost_history_mean.append(np.mean(costs))

                print(
                    f"Gen {gen:03d} | "
                    f"min = {self.cost_history_min[-1]}, "
                    f"mean = {self.cost_history_mean[-1]}"
                )

        except KeyboardInterrupt:

            print("\nOptimization interrupted by user")

        finally:
            pool.close()
            pool.join()

        # ===== best solution =====
        best = tools.selBest(pop, 1)[0]

        self.best_params = expand_params(
            np.array(best),
            self.muscles,
            symmetry=symmetry
        )

        plot_cost_history(self.model_name, self.cost_history_mean, "mean")
        plot_cost_history(self.model_name, self.cost_history_min, "min")

        return {
            "params": [self.best_params],
            "fitness": [tuple(best.fitness.values)]
        }