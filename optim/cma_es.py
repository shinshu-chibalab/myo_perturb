import signal
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

def _init_worker_safe(*args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore SIGINT in worker processes
    _init_worker(*args)  # Call the original initializer

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

        worker = partial(
            run_simulation_worker, 
            controller=self.controller, 
            evaluator=self.evaluator, 
            delay_time=delay_time, 
            noise_std=noise_std
        )

        best_x = None
        best_fitness = None
        best_cost = np.inf

        pool = None
        interrupt = False

        try:
            pool = Pool(
                processes=n_jobs,
                initializer=_init_worker_safe,
                initargs=(
                    self.model_path,
                    self.muscles,
                    self.sim_steps,
                    self.model.qpos0.copy(),
                    self.com_init_height,
                    symmetry,
                )
            )

            while not es.stop():
                solutions = es.ask()
                
                eval_results = pool.map(worker, solutions, chunksize=4)
                costs = [float(res[0]) for res in eval_results]

                best_idx = int(np.argmin(costs))
                generation_best_cost = costs[best_idx]

                if generation_best_cost < best_cost:

                    best_cost = generation_best_cost

                    best_x = np.asarray(
                        solutions[best_idx],
                        dtype=float,
                    ).copy()

                    best_fitness = tuple(
                        eval_results[best_idx]
                    )


                es.tell(solutions, costs)
                es.disp()

                self.cost_history_min.append(float(np.min(costs)))
                self.cost_history_mean.append(float(np.mean(costs)))

            # best_x = es.result.xbest
            # best_fitness = pool.map(worker, [best_x], chunksize=1)[0]

        except KeyboardInterrupt:
            interrupt = True
            print("\n[INFO] Optimization interrupted by user.", flush=True)

        finally:
            if pool is not None:
                if interrupt:
                    pool.terminate()
                else:
                    pool.close()
                
                pool.join()

            print("[INFO] Worker processes terminated.", flush=True)

        if best_x is None:
            print("[WARNING] No completed generations. " "No results to save.", flush=True)
            return {
                "params": [],
                "fitness": [],
            }
        
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