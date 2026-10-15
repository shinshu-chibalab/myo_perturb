import numpy as np
from cma import CMAEvolutionStrategy
from functools import partial
from multiprocessing import Pool

from mujoco import MjModel, MjData, mj_forward
import mujoco

from utils.param_utils import create_x0_bounds, expand_params
from optim.base import OptimizerBase
from simulation.worker import run_simulation_worker, _init_worker
from render.plot_cost import plot_cost_history

class CMA_ES(OptimizerBase):
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

        self.sim_steps = sim_steps

        self.model_name = model_name

        self.cost_history_min = []
        self.cost_history_mean = []

    def optimize(self, sigma0=1.0, maxiter=50, popsize=100, delay_time=0.0, noise_std=0.0, n_jobs=1, symmetry=True):

        x0, bounds_lower, bounds_upper = create_x0_bounds(self.muscles, symmetry=symmetry)

        self.delay_time = delay_time
        self.noise_std = noise_std

        opts = {
            "maxiter": maxiter,
            "popsize": popsize,
            "CMA_stds": (bounds_upper - bounds_lower),
            "tolfun": 1e-12,
            "tolx": 1e-12,
            "tolfunhist": 1e-12,
            "bounds": [bounds_lower, bounds_upper]
        }

        es = CMAEvolutionStrategy(x0, sigma0, opts)

        print(f"Using {n_jobs} parallel workers")

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

        worker = partial(
            run_simulation_worker, 
            controller=self.controller, 
            evaluator=self.evaluator, 
            delay_time=delay_time, 
            noise_std=noise_std
        )

        try:
            while not es.stop():
                solutions = es.ask()
                
                eval_results = pool.map(worker, solutions, chunksize=4)
                costs = [float(res[0]) for res in eval_results]

                es.tell(solutions, costs)
                es.disp()

                self.cost_history_min.append(float(np.min(costs)))
                self.cost_history_mean.append(float(np.mean(costs)))

            best_x = es.result.xbest
            best_fitness = pool.map(worker, [best_x], chunksize=1)[0]

        finally:
            pool.close()
            pool.join()
        
        plot_cost_history(self.model_name, self.cost_history_mean, "mean")
        plot_cost_history(self.model_name, self.cost_history_min, "min")

        best_params = expand_params(
            best_x,
            self.muscles,
            symmetry=symmetry,
        )

        return {
            "params": [best_params],
            "fitness": [tuple(best_fitness)],
        }