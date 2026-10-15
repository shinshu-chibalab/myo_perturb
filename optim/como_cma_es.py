import os
import numpy as np
import comocma
import myo_sim
from functools import partial
from multiprocessing import Pool

from mujoco import MjModel, MjData, mj_forward

from utils.param_utils import create_x0_bounds, expand_params
from optim.base import OptimizerBase
from simulation.worker import run_simulation_worker, _init_worker
from render.plot_cost import plot_cost_history
from render.plot_pareto import plot_pareto_front

class _Fitness:
    def __init__(self, values):
        self.values = tuple(values)


class _IndividualLike:
    def __init__(self, fitness_values):
        self.fitness = _Fitness(fitness_values)


class COMO_CMA_ES(OptimizerBase):
    def __init__(self, model, muscles, controller, evaluator, sim_steps, model_name):
        self.model_path = model_path
        self.model, self.data = myo_sim.load_model(model)

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

    @staticmethod
    def nondominated_indices(F):
        """
        2目的最小化専用の高速な非劣解抽出。
        F[:, 0] = f1, F[:, 1] = f2
        """
        F = np.asarray(F, dtype=float)

        if F.ndim != 2 or F.shape[1] != 2:
            raise ValueError("This fast nondominated_indices is only for 2 objectives.")

        # f1昇順、同じf1ならf2昇順
        order = np.lexsort((F[:, 1], F[:, 0]))

        nd = []
        best_f2 = np.inf

        for idx in order:
            f2 = F[idx, 1]

            # f1が小さい順に見ているので、
            # これまでのbest_f2より小さければ非劣解
            if f2 < best_f2:
                nd.append(idx)
                best_f2 = f2

        return np.array(nd, dtype=int)

    @staticmethod
    def select_diverse_pareto(X, F, n_select=10):
        if len(X) <= n_select:
            return np.arange(len(X))

        f_min = F.min(axis=0)
        f_max = F.max(axis=0)
        Fn = (F - f_min) / (f_max - f_min + 1e-12)

        idx_min_f1 = np.argmin(Fn[:, 0])
        idx_min_f2 = np.argmin(Fn[:, 1])

        selected = list(dict.fromkeys([idx_min_f1, idx_min_f2]))
        remaining = list(set(range(len(X))) - set(selected))

        while len(selected) < n_select and remaining:
            best_idx = None
            best_dist = -1.0

            for i in remaining:
                d = min(np.linalg.norm(Fn[i] - Fn[j]) for j in selected)

                if d > best_dist:
                    best_dist = d
                    best_idx = i

            selected.append(best_idx)
            remaining.remove(best_idx)

        return np.array(selected, dtype=int)

    def save_pareto_front_npy(self, X, F):
        save_path = os.path.join("results", self.model_name, "pareto_front_comocma.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        data = []
        for x, f in zip(X, F):
            data.append((float(f[0]), float(f[1]), np.asarray(x, dtype=float)))

        np.save(save_path, np.array(data, dtype=object), allow_pickle=True)

    def optimize(self, sigma0=1.0, maxiter=50, popsize=100, delay_time=0.0, noise_std=0.0, n_jobs=1, symmetry=True, n_kernels=10, reference_point=(1e6, 1e6),):
        x0, bounds_lower, bounds_upper, *_ = create_x0_bounds(self.muscles, symmetry=symmetry)

        dim = len(x0)
        kernel_popsize = max(4, popsize // n_kernels)

        # ===== initial kernels =====
        x_starts = [np.clip(x0, bounds_lower, bounds_upper).tolist()]

        for _ in range(n_kernels - 1):
            if np.random.rand() < 0.5:
                x = np.random.uniform(bounds_lower, bounds_upper)
            else:
                x = x0 * (1.0 + np.random.uniform(-0.05, 0.05, size=dim))
                x = np.clip(x, bounds_lower, bounds_upper)

            x_starts.append(x.tolist())

        sigma_starts = [1.0] * n_kernels

        cma_opts = {
            "bounds": [bounds_lower, bounds_upper],
            "popsize": kernel_popsize,
            "verb_disp": 0,
            "CMA_stds": (bounds_upper - bounds_lower) * sigma0,
            "maxiter": maxiter,
            "tolfun": 1e-30,
            "tolx": 1e-30,
            "tolfunhist": 1e-30,
        }

        kernels = comocma.get_cmas(
            x_starts,
            sigma_starts,
            inopts=cma_opts,
        )

        moes = comocma.Sofomore(
            kernels,
            reference_point=list(reference_point),
            opts={
                "archive": True
            },
        )

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
            noise_std=noise_std,
        )

        archive_X = []
        archive_F = []

        try:
            for gen in range(maxiter):
                solutions = moes.ask("all")

                solutions = [
                    np.clip(np.asarray(x, dtype=float), bounds_lower, bounds_upper)
                    for x in solutions
                ]

                fitnesses = pool.map(worker, solutions, chunksize=4)
                fitnesses = np.asarray(fitnesses, dtype=float)

                moes.tell(
                    [x.tolist() for x in solutions],
                    fitnesses.tolist(),
                )

                archive_X.extend([x.copy() for x in solutions])
                archive_F.extend([f.copy() for f in fitnesses])

                f1 = fitnesses[:, 0]
                f2 = fitnesses[:, 1]

                self.cost_history_min1.append(float(np.min(f1)))
                self.cost_history_min2.append(float(np.min(f2)))
                self.cost_history_mean1.append(float(np.mean(f1)))
                self.cost_history_mean2.append(float(np.mean(f2)))

                best_f1_so_far = np.min(np.asarray(archive_F)[:, 0])
                best_f2_so_far = np.min(np.asarray(archive_F)[:, 1])

                print(
                    f"Gen {gen:03d} | "
                    f"evals={len(solutions)} | "
                    f"f1[min]={np.min(f1)}, f2[min]={np.min(f2)}"
                )

                if gen % 100 == 0:
                    print(f"best_f1={best_f1_so_far}, best_f2={best_f2_so_far}")

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")

        finally:
            pool.close()
            pool.join()

        archive_X = np.asarray(archive_X, dtype=float)
        archive_F = np.asarray(archive_F, dtype=float)

        nd_idx = self.nondominated_indices(archive_F)
        pareto_X = archive_X[nd_idx]
        pareto_F = archive_F[nd_idx]

        self.save_pareto_front_npy(pareto_X, pareto_F)

        selected_idx = self.select_diverse_pareto(
            pareto_X,
            pareto_F,
            n_select=10,
        )

        selected_X = pareto_X[selected_idx]
        selected_F = pareto_F[selected_idx]

        selected_params = [
            expand_params(x, self.muscles, symmetry=symmetry)
            for x in selected_X
        ]

        print(f"selected pareto solutions: {len(selected_params)}")

        for i, fit in enumerate(selected_F, start=1):
            print(f"selected {i}: f1={fit[0]}, f2={fit[1]}")

        plot_cost_history(self.model_name, self.cost_history_mean1, "mean_energy")
        plot_cost_history(self.model_name, self.cost_history_min1, "min_energy")
        plot_cost_history(self.model_name, self.cost_history_mean2, "mean_com")
        plot_cost_history(self.model_name, self.cost_history_min2, "min_com")

        pareto_front_like = [
            _IndividualLike(fit)
            for fit in pareto_F
        ]

        plot_pareto_front(self.model_name, [pareto_front_like])

        return {
            "params": selected_params,
            "fitness": [
                tuple(fit)
                for fit in selected_F
            ],
        }