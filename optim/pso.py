import numpy as np
import pyswarms as ps
from pyswarms.backend.operators import compute_pbest, compute_objective_function
from functools import partial
from multiprocessing import Pool

from mujoco import MjModel, MjData, mj_forward
import mujoco

from utils.param_utils import create_x0_bounds, expand_params
from optim.base import OptimizerBase
from simulation.worker import run_simulation_worker, _init_worker
from render.plot_cost import plot_cost_history

class PSO(OptimizerBase):
    def __init__(self, model_path, muscles, controller, evaluator, sim_steps, model_name):
        self.model_path = model_path
        self.model = MjModel.from_xml_path(model_path)
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

    @staticmethod
    def normalize(x, lower, upper):
        return (x - lower) / (upper - lower + 1e-12)

    @staticmethod
    def denormalize(z, lower, upper):
        return lower + z * (upper - lower)

    def pso(self, objective_func, maxiter, n_particles, dimensions, x0_norm):
        bounds = (
            np.zeros(dimensions),  # lower bounds
            np.ones(dimensions),   # upper bounds
        )

        init_pos = np.random.normal(
            loc=x0_norm,
            scale=0.1,     # 初期ばらつき（適宜調整）
            size=(n_particles, dimensions)
        )

        init_pos = np.clip(init_pos, bounds[0], bounds[1])

        start_opts = {'c1':2.5, 'c2':0.5, 'w':0.9}
        end_opts= {'c1':0.5, 'c2':2.5, 'w':0.4}     # Ref:[1]
        oh_strategy={ "w":'exp_decay', "c1":'nonlin_mod',"c2":'lin_variation'}

        opt = ps.single.GlobalBestPSO(
            n_particles=n_particles, 
            dimensions=dimensions, 
            options=start_opts, 
            oh_strategy=oh_strategy, 
            bounds=bounds, 
            init_pos=init_pos
        )

        swarm = opt.swarm
        opt.bh.memory = swarm.position
        opt.vh.memory = swarm.position
        swarm.pbest_cost = np.full(opt.swarm_size[0], np.inf)

        for i in range(maxiter):
            # Compute cost for current position and personal best
            swarm.current_cost =  compute_objective_function(swarm, objective_func)
            swarm.pbest_pos, swarm.pbest_cost = compute_pbest(swarm)
            swarm.best_pos, swarm.best_cost = opt.top.compute_gbest(swarm)

            # Perform options update
            swarm.options = opt.oh(
                opt.options, 
                iternow=i, 
                itermax=maxiter, 
                end_opts=end_opts
            )

            self.cost_history_min.append(float(np.min(swarm.current_cost)))
            self.cost_history_mean.append(float(np.mean(swarm.current_cost)))

            if i % 100 == 0:
                print("Iteration:", i," Options: ", swarm.options)    # print to see variation
                print(f"[iter {i:04d}] best_cost = {swarm.best_cost} mean_cost = {self.cost_history_mean[-1]}")
            
            # Perform velocity and position updates
            swarm.velocity = opt.top.compute_velocity(
                swarm, 
                opt.velocity_clamp, 
                opt.vh, 
                opt.bounds,
            )

            swarm.position = opt.top.compute_position(
                swarm, 
                opt.bounds, 
                opt.bh,
            )

        # Obtain the final best_cost and the final best_position
        best_cost = float(swarm.best_cost)
        best_pos_norm = swarm.best_pos.copy()

        return best_cost, best_pos_norm


    def optimize(self, sigma0=1, maxiter=50, popsize=100, delay_time=0.0, noise_std=0.0, n_jobs=1, symmetry=True):

        # ===== 初期パラメータ =====
        x0, bounds_lower, bounds_upper = create_x0_bounds(self.muscles, symmetry=symmetry)
        x0_norm = self.normalize(x0, bounds_lower, bounds_upper)

        dim = len(x0)
        
        print(f"Using {n_jobs} parallel workers")

        # ===== Worker初期化 =====
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
            ),
        )

        worker = partial(
            run_simulation_worker, 
            controller=self.controller,
            evaluator=self.evaluator,
            delay_time=delay_time,
            noise_std=noise_std
        )

        def objective_func(Z):
            # Z.shape = (n_particles, dim), normalized [0, 1]
            X = self.denormalize(Z, bounds_lower, bounds_upper)

            eval_results = pool.map(worker, X, chunksize=4)

            costs = np.array(
                [float(res[0]) for res in eval_results],
                dtype=float,
            )

            return costs

        try:
            best_cost, best_norm = self.pso(
                objective_func=objective_func,
                maxiter=maxiter,
                n_particles=popsize,
                dimensions=dim,
                x0_norm=x0_norm,
            )

            best_x = self.denormalize(
                best_norm,
                bounds_lower,
                bounds_upper,
            )

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

        print("Best cost = ", best_cost)
        print("Best position = ", best_fitness)

        return {
            "params": [best_params],
            "fitness": [tuple(best_fitness)],
        }