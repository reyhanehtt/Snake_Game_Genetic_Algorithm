# Snake GAME (Neuroevolution Snake)

Train a simple neural policy (3-layer MLP) for the classic Snake game using a Genetic Algorithm (GA).

## Features
- Headless training using Pygame's dummy video driver
- Clean engine (`snakegame/game.py`) with a grid-based world (30×30, 10px cells)
- Compact MLP policy (7→9→15→3) with softmax output over **left/straight/right** relative turns
- Simple GA with elitism, uniform crossover, and Gaussian mutation
- Reproducible seeds & CLI scripts

## Quickstart

```bash
# (Recommended) Create & activate a venv, then:
pip install -r requirements.txt

# Train for a few generations headless
python -m scripts.train --generations 10 --population 30 --headless

# Play the best genome (renders a window)
python -m scripts.play --weights runs/best_genome.npy
