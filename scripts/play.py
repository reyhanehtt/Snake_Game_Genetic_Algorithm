import argparse, numpy as np, pygame
from snakega.model import NUM_WEIGHTS
from snakega.fitness import run_game_with_policy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='Path to .npy genome file')
    args = ap.parse_args()

    genome = np.load(args.weights).astype(np.float32)
    assert genome.size == NUM_WEIGHTS, f"Expected {NUM_WEIGHTS} weights, got {genome.size}"

    print("Playing a single game with rendering. Close the window to exit.")
    fit, tie = run_game_with_policy(genome, headless=False)
    print(f"Finished. Fitness={fit}, tie={tie}")

if __name__ == '__main__':
    main()
