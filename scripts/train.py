# scripts/train.py
import os, argparse, numpy as np
from snakega.ga import GAConfig, train
from snakega.model import NUM_WEIGHTS, random_genome

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--generations', type=int)
    p.add_argument('--population', type=int)
    p.add_argument('--headless', action='store_true', help='No window')
    p.add_argument('--outdir', type=str, default='runs')
    p.add_argument('--eval-games', type=int, help='Games per genome (averaged)')
    p.add_argument('--jobs', type=int, help='Workers (0=all cores, 1=sequential)')
    p.add_argument('--patience', type=int, default=0, help='Early stop after N stagnant gens')
    p.add_argument('--resume', type=str, help='Warm-start from an existing .npy genome')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = GAConfig(outdir=args.outdir)
    if args.generations: cfg.generations = args.generations
    if args.population:  cfg.population  = args.population
    if args.headless:    cfg.headless    = True
    if args.eval_games:  cfg.eval_games  = args.eval_games
    if args.jobs is not None:       cfg.n_jobs = args.jobs
    if args.patience is not None:   cfg.patience = args.patience

    # Optional: warm start population with a known good genome
    if args.resume:
        seed = np.load(args.resume).astype(np.float32)
        assert seed.size == NUM_WEIGHTS, f"Expected {NUM_WEIGHTS} weights, got {seed.size}"
        rng = np.random.default_rng(cfg.seed)
        pop = [seed.copy()]
        while len(pop) < cfg.population:
            # alternate jittered copies and fresh genomes
            if len(pop) % 2 == 0:
                g = seed + rng.normal(0, cfg.mutation_std, size=seed.shape).astype(np.float32)
            else:
                g = random_genome(rng)
            pop.append(g)
        warm = os.path.join(cfg.outdir, "_warmstart.npy")
        np.save(warm, np.stack(pop, axis=0))
        print(f"Warm-start population saved to {warm} — train() will pick it up if compatible.")

    best, score = train(cfg)
    out = os.path.join(args.outdir, 'best_genome.npy')
    np.save(out, best)
    print(f"Saved best genome to {out} (fitness={score:.1f})")

if __name__ == '__main__':
    main()
