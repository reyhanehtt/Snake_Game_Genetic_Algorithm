import os
import csv
import multiprocessing as mp
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from . import config as C
from .model import NUM_WEIGHTS, random_genome
from .fitness import run_game_with_policy

@dataclass
class GAConfig:
    population: int = C.POPULATION
    parents: int = C.PARENTS
    generations: int = C.GENERATIONS
    mutation_rate: float = C.MUTATION_RATE
    mutation_std: float = C.MUTATION_STD
    headless: bool = True
    steps_per_game: int = C.STEPS_PER_GAME
    seed: int = C.RANDOM_SEED
    eval_games: int = C.EVAL_GAMES
    n_jobs: int = C.N_JOBS
    outdir: str = "runs"
    patience: int = 0  # early stop if no improvement for N gens (0 = off)

def _eval_one(args):
    genome, headless, eval_games, base_seed = args
    rng = np.random.default_rng(base_seed)
    fits, ties = [], []
    for _ in range(eval_games):
        s = int(rng.integers(0, 2**31 - 1))
        fit, tie = run_game_with_policy(genome, headless=headless, max_steps=None, rng_seed=s)
        fits.append(fit)
        ties.append(tie)
    return float(np.mean(fits)), float(np.mean(ties))

def evaluate_population(pop, headless=True, eval_games=1, seed=0, n_jobs=1) -> Tuple[np.ndarray, np.ndarray]:
    args = [(pop[i], headless, eval_games, seed + i) for i in range(pop.shape[0])]
    if n_jobs == 0:
        n_jobs = os.cpu_count() or 1
    if n_jobs > 1:
        with mp.Pool(processes=n_jobs, maxtasksperchild=1) as pool:
            res = pool.map(_eval_one, args)
    else:
        res = list(map(_eval_one, args))
    f1 = np.array([r[0] for r in res], dtype=np.float32)
    f2 = np.array([r[1] for r in res], dtype=np.float32)
    return f1, f2

def select_parents(pop, f1, f2, k):
    order = np.lexsort((-f2, -f1))
    return pop[order][:k].copy(), order[:k], order

def uniform_crossover(rng, parents, n_offspring):
    off = np.empty((n_offspring, parents.shape[1]), dtype=np.float32)
    num_par = parents.shape[0]
    for i in range(n_offspring):
        p1, p2 = rng.integers(0, num_par, size=2)
        while p2 == p1:
            p2 = rng.integers(0, num_par)
        mask = rng.random(size=(parents.shape[1],)) < 0.5
        off[i] = np.where(mask, parents[p1], parents[p2])
    return off

def mutate(rng, off, rate, std):
    mask = rng.random(size=off.shape) < rate
    noise = rng.normal(0.0, std, size=off.shape).astype(np.float32)
    return off + mask * noise

def train(cfg: GAConfig):
    os.makedirs(cfg.outdir, exist_ok=True)
    metrics_path = os.path.join(cfg.outdir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["generation", "best_fitness", "mean_fitness", "mutation_std"])

    rng = np.random.default_rng(cfg.seed)
    pop = np.stack([random_genome(rng) for _ in range(cfg.population)], axis=0)

    best = None
    best_score = -1e9
    cur_std = cfg.mutation_std
    stagnation = 0

    for g in range(cfg.generations):
        f1, f2 = evaluate_population(pop, headless=cfg.headless, eval_games=cfg.eval_games,
                                     seed=cfg.seed + g * 10000, n_jobs=cfg.n_jobs)
        parents, parent_idx, order = select_parents(pop, f1, f2, cfg.parents)

        # Logging
        mean_fit = float(np.mean(f1))
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([g+1, float(np.max(f1)), mean_fit, cur_std])

        # Save top-5 genomes of this gen
        topk = min(5, pop.shape[0])
        for r in range(topk):
            np.save(os.path.join(cfg.outdir, f"gen{g+1:03d}_top{r+1}.npy"), pop[order[r]])

        # Elitism
        elite_n = min(2, parents.shape[0])
        elite = parents[:elite_n]

        # Offspring
        need = cfg.population - elite_n
        children = uniform_crossover(rng, parents, need)
        children = mutate(rng, children, cfg.mutation_rate, cur_std)
        pop = np.vstack([elite, children])

        # Best tracking + adaptive std
        gen_best_idx = int(np.argmax(f1))
        gen_best_fit = float(f1[gen_best_idx])
        if gen_best_fit > best_score + 1e-6:
            best_score = gen_best_fit
            best = pop[gen_best_idx].copy()
            cur_std = min(cur_std * 1.02, 0.5)
            stagnation = 0
        else:
            cur_std = max(cur_std * 0.98, 0.02)
            stagnation += 1

        print(f"Gen {g+1}/{cfg.generations} | best_fitness={best_score:.1f} | mut_std={cur_std:.3f} | mean={mean_fit:.1f}")

        if cfg.patience and stagnation >= cfg.patience:
            print(f"Early stopping: no improvement for {cfg.patience} generations.")
            break

    return best, best_score
